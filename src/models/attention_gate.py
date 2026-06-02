"""
AttentionGate — learned per-particle scalar gate for JEPA pretraining.

Computes a soft importance weight for each particle in the jet using the
Lorentz-invariant pairwise interaction matrix U produced by ParticleProcessor.
The gate is applied multiplicatively to the particle multivector embeddings
before they enter the context encoder, allowing the model to downweight
particles that are geometrically irrelevant to the masked position.

Physics motivation:
  Not all particles are equally informative for predicting a masked particle's
  embedding. A b-quark is more relevant to predicting another b-quark than a
  soft wide-angle QCD emission. The gate learns this from the training signal
  (JEPA loss) using the kt-clustering variables (ln Δ, ln kT, ln z, ln m²)
  that encode the geometric structure of jet splittings.

Reference:
  Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018 — canonical paper
  for scalar feature gating in deep networks (applied here to the particle
  sequence dimension rather than channels).
"""

import torch.nn as nn
from torch import Tensor


class AttentionGate(nn.Module):
    """
    Learned per-particle scalar gate derived from pairwise interaction features.

    For each particle i, aggregates its pairwise relationships with all other
    particles by mean-pooling U over the neighbor dimension, then maps the
    resulting 4-dim summary through a small MLP to produce a scalar gate in
    [0, 1]. Padding pairs (set to -1e9 in the processor) are included in the
    mean but have negligible effect due to their large negative values pulling
    the log features down uniformly.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the MLP (default 16).
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()

        # ── MLP: (B, N, 4) → (B, N, 1) ──────────────────────────────────────
        # Two hidden layers with ReLU; sigmoid output constrains gate to [0, 1]
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),        # project interaction features up
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),        # collapse to scalar per particle
            nn.Sigmoid()                     # gate ∈ [0, 1]
        )

    def forward(self, U: Tensor) -> Tensor:
        """
        Parameters
        ----------
        U : Tensor, shape (B, N, N, 4)
            Pairwise interaction matrix from ParticleProcessor.
            Features along dim 3: (ln_delta, ln_kT, ln_z, ln_m2).
            Padding pairs are filled with -1e9; diagonal is 0.

        Returns
        -------
        gate : Tensor, shape (B, N, 1)
            Per-particle scalar importance weights in [0, 1].
            Applied multiplicatively to particle embeddings before the encoder.
        """
        # ── Aggregate pairwise features per particle ──────────────────────────
        # Mean over neighbor dimension j → each particle i gets a 4-dim summary
        # of its average geometric relationship with the rest of the jet
        avg = U.mean(dim=2)         # (B, N, N, 4) → (B, N, 4)

        # ── Map to scalar gate ────────────────────────────────────────────────
        return self.mlp(avg)        # (B, N, 4) → (B, N, 1)
