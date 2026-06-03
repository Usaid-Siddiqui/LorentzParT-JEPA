"""
Embedding loss for JEPA pretraining.

Computes MSE between LayerNorm-normalised prediction and target embeddings.
Normalising both vectors before the MSE loss is the standard JEPA trick to
prevent representation collapse: without it, the model can trivially minimise
the loss by mapping everything to the origin.

The return signature (total_loss, (total_loss,)) matches the (loss, components)
convention used by MaskedModelTrainer so JEPATrainer can reuse the same
checkpointing and logging infrastructure.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EmbeddingLoss(nn.Module):
    """
    LayerNorm-normalised MSE loss for JEPA embedding prediction.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding vectors (default 128).
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Separate norms so prediction and target are independently normalised.
        # Using fixed (non-learnable) LayerNorm: we want unit-variance embeddings,
        # not learnable scale/shift that could absorb the loss signal.
        self.norm_pred = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.norm_target = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(
        self, pred: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Parameters
        ----------
        pred : Tensor, shape (B, K, embed_dim) or (B, embed_dim)
            Predictor output for the masked particle positions.
        target : Tensor, shape (B, K, embed_dim) or (B, embed_dim)
            Target encoder embeddings at the masked positions (detached).

        Returns
        -------
        loss : Tensor, scalar
        components : tuple of Tensor
            Single-element tuple (loss,) for compatibility with trainer logging.
        """
        # Flatten (B, K, D) → (B*K, D) so LayerNorm and MSE work over all K positions
        pred   = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        pred_norm   = self.norm_pred(pred)
        target_norm = self.norm_target(target)
        loss = F.mse_loss(pred_norm, target_norm)
        return loss, (loss,)
