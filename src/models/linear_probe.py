from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .lorentz_part import LorentzParTEncoder
from .processor import ParticleProcessor


class LinearProbeModel(nn.Module):
    """
    Frozen pretrained encoder + mean-pool + linear classification head.

    Implements the standard SSL linear evaluation protocol: only the head
    is trained; the encoder is frozen throughout. The encoder output at
    each valid (non-padding) particle position is mean-pooled to produce
    a single jet-level embedding, which is passed to the linear head.

    Parameters
    ----------
    encoder_weights : str
        Path to a checkpoint saved by JEPATrainer or MaskedModelTrainer.
        Must contain keys with the 'encoder.*' prefix (standard convention).
    embed_dim : int
        Encoder output dimension (default 128).
    num_classes : int
        Number of output classes (default 10).
    encoder_kwargs : dict, optional
        Forwarded to LorentzParTEncoder.__init__ (num_heads, num_layers, etc.).
    """

    def __init__(
        self,
        encoder_weights: str,
        embed_dim: int = 128,
        num_classes: int = 10,
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        kw = encoder_kwargs or {}
        self.processor = ParticleProcessor(to_multivector=True)
        self.encoder = LorentzParTEncoder(embed_dim=embed_dim, **kw)

        state_dict = torch.load(encoder_weights, map_location='cpu', weights_only=True)
        filtered = {
            k[len('encoder.'):]: v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
        self.encoder.load_state_dict(filtered, strict=False)

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, N, 4)
            Normalized particle features [pT, eta, phi, E].

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
        """
        padding_mask = (x[..., 3] == 0).float()  # (B, N)

        with torch.no_grad():
            mv, U = self.processor(x)
            embeddings = self.encoder(mv, padding_mask, U)  # (B, N, embed_dim)

        # Mean pool over valid (non-padding) particles
        valid = 1.0 - padding_mask                                    # (B, N)
        valid_sum = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        jet_embed = (embeddings * valid.unsqueeze(-1)).sum(dim=1) / valid_sum  # (B, embed_dim)

        return self.head(jet_embed)
