"""
ParticleJEPA — Joint-Embedding Predictive Architecture for jet particle clouds.

Like the original MAE in this codebase, we mask one particle at a time.
The key difference is *what* gets predicted:

  MAE:  reconstructs raw 4-vector features (pT, η, φ, E) of the masked particle
  JEPA: predicts the target encoder's latent embedding at the masked position

This forces the context encoder to learn about the *physics* of particle
relationships rather than memorising low-level feature statistics.

Architecture:
  attention_gate   — small MLP; derives per-particle scalar weights from U
                     and applies them to multivector embeddings before the encoder
  context_encoder  — trainable LorentzParTEncoder; sees gated, zeroed masked particle
  target_encoder   — EMA copy of context_encoder (frozen); sees full input
  predictor        — narrow bottleneck transformer; maps context sequence →
                     predicted embedding at the masked position

The target encoder is updated as a slow-moving average of the context encoder
after each optimisation step, providing a stable, progressively improving
training signal without collapsing to trivial solutions.

Reference:
  Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture", CVPR 2023.
"""

import copy
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .lorentz_part import LorentzParTEncoder
from .predictor import ParticlePredictor
from .processor import ParticleProcessor
from .attention_gate import AttentionGate


class ParticleJEPA(nn.Module):
    """
    Particle JEPA for self-supervised pretraining on jet particle clouds.

    Parameters
    ----------
    embed_dim : int
        Encoder embedding dimension (default 128).
    num_heads : int
        Number of attention heads in the encoder (default 8).
    num_layers : int
        Number of encoder transformer blocks (default 8).
    dropout : float
        Dropout rate for the encoder (default 0.1).
    expansion_factor : int
        FFN expansion factor in encoder blocks (default 4).
    pair_embed_dims : list of int
        Hidden dimensions for the pairwise interaction embedding MLP.
    predictor_dim : int
        Bottleneck hidden dimension for the predictor (default 64).
    predictor_heads : int
        Attention heads in the predictor transformer (default 4).
    predictor_layers : int
        Number of transformer blocks in the predictor (default 4).
    predictor_dropout : float
        Dropout rate for the predictor (default 0.1).
    max_num_particles : int
        Maximum particles per jet; used for predictor positional embedding.
    ema_momentum : float
        Starting EMA momentum for target encoder update (default 0.996).
        Values close to 1 → slow update (stable target).
    use_attention_gate : bool
        If True, applies a learned per-particle scalar gate (derived from U)
        to the context encoder's multivector inputs before encoding (default True).
        Set to False to ablate the gating and run vanilla JEPA.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        pair_embed_dims: Optional[list] = None,
        predictor_dim: int = 64,
        predictor_heads: int = 4,
        predictor_layers: int = 4,
        predictor_dropout: float = 0.1,
        max_num_particles: int = 128,
        ema_momentum: float = 0.996,
        use_attention_gate: bool = True,
    ):
        super().__init__()

        if pair_embed_dims is None:
            pair_embed_dims = [64, 64, 64]

        self.ema_momentum = ema_momentum
        self.use_attention_gate = use_attention_gate

        # Shared processor: computes multivectors + pairwise interaction features
        self.processor = ParticleProcessor(to_multivector=True)

        # Context encoder (trainable)
        self.context_encoder = LorentzParTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            expansion_factor=expansion_factor,
            pair_embed_dims=pair_embed_dims,
        )

        # Attention gate: soft per-particle importance weighting from U
        # Instantiated regardless; only applied when use_attention_gate=True
        self.attention_gate = AttentionGate()

        # Target encoder (frozen; updated via EMA after each step)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Predictor: maps context encoder output → predicted target embedding
        self.predictor = ParticlePredictor(
            encoder_dim=embed_dim,
            predictor_dim=predictor_dim,
            num_heads=predictor_heads,
            num_layers=predictor_layers,
            max_num_particles=max_num_particles,
            dropout=predictor_dropout,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        """
        Exponential moving average update of the target encoder.

            θ_target = momentum * θ_target + (1 - momentum) * θ_context

        Call this after every optimiser step during pretraining.

        Parameters
        ----------
        momentum : float, optional
            EMA momentum to use this step. If None, uses self.ema_momentum.
        """
        if momentum is None:
            momentum = self.ema_momentum

        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_param.data.lerp_(ctx_param.data, 1.0 - momentum)

    def forward(
        self, x: Tensor, mask_idx: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, N, 4)
            Particle features [pT, η, φ, E] for each jet, zero-padded.
        mask_idx : Tensor, shape (B,), dtype long
            Index of the particle to mask in each jet.

        Returns
        -------
        predicted_embed : Tensor, shape (B, embed_dim)
            Predictor's output for the masked position.
        target_embed : Tensor, shape (B, embed_dim)
            Target encoder's embedding at the masked position (detached).
        """
        B, N, F = x.shape
        device = x.device
        batch_idx = torch.arange(B, device=device)

        # ---------- Normalise mask_idx to (B, K) ----------
        # Trainer squeezes (B, 1) → (B,) for K=1; unsqueeze back for uniform handling
        if mask_idx.dim() == 1:
            mask_idx = mask_idx.unsqueeze(1)   # (B,) → (B, 1)
        K = mask_idx.shape[1]                  # number of masked particles

        # ---------- Build padding masks ----------
        # Valid particles: energy > 0  →  padding_mask = 0 (attend)
        # Padding particles: energy == 0  →  padding_mask = 1 (ignore)
        full_padding_mask = (x[..., 3] == 0).float()   # (B, N)

        # Zero out all K masked particles; keep their padding_mask = 0
        # so the encoder knows a particle exists at those positions
        masked_x = x.clone()
        context_padding_mask = full_padding_mask.clone()
        for k in range(K):
            masked_x[batch_idx, mask_idx[:, k]] = 0.0
            context_padding_mask[batch_idx, mask_idx[:, k]] = 0.0

        # ---------- Process inputs ----------
        full_mv, U_full = self.processor(x)
        context_mv, U_context = self.processor(masked_x)

        # ---------- Attention gate (optional) ----------
        if self.use_attention_gate:
            gate = self.attention_gate(U_context)   # (B, N, 1)
            context_mv = context_mv * gate          # (B, N, 16)

        # ---------- Target encoder (no gradient) ----------
        with torch.no_grad():
            target_out = self.target_encoder(full_mv, full_padding_mask, U_full)
            # Extract K target embeddings per jet: (B, K, embed_dim)
            target_embed = target_out[batch_idx.unsqueeze(1), mask_idx]

        # ---------- Context encoder (trainable) ----------
        context_out = self.context_encoder(context_mv, context_padding_mask, U_context)

        # ---------- Predictor ----------
        predicted_embed = self.predictor(context_out, mask_idx)   # (B, K, embed_dim)

        return predicted_embed, target_embed.detach()
