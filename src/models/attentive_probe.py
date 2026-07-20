from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .classifier import ClassAttentionBlock, Classifier
from .lorentz_part import LorentzParTEncoder
from .processor import ParticleProcessor


class AttentiveProbeModel(nn.Module):
    """
    Frozen pretrained encoder + trainable class-attention head.

    The attentive-probe evaluation protocol (I-JEPA style): the encoder is
    frozen, and a small class-attention head — the exact head LorentzParT uses
    for classification (a learned CLS token attending over the particle
    embeddings) — is trained on top. It sits between the linear probe (frozen
    encoder + mean-pool + linear) and full fine-tuning: strictly more expressive
    than mean-pooling, but the encoder never updates. Linear probes are known to
    undersell JEPA features, so this is the fair frozen-encoder ceiling.

    Mirrors ``LinearProbeModel``: same ``encoder.*``-prefixed checkpoint loading
    and the same encoder freeze; only the head differs (class-attention instead
    of mean-pool + linear).

    Parameters
    ----------
    encoder_weights : str, optional
        Path to a checkpoint saved by JEPATrainer or MaskedModelTrainer.
        Must contain keys with the 'encoder.*' prefix (standard convention).
        If None, the encoder is left randomly initialized (still frozen) — the
        random-feature control for the frozen-probe protocols.
    embed_dim : int
        Encoder output dimension (default 128).
    num_classes : int
        Number of output classes (default 10).
    num_heads : int
        Attention heads in the class-attention head (default 8).
    num_cls_layers : int
        Number of class-attention blocks in the head (default 2).
    hidden_dim, num_mlp_layers, expansion_factor, dropout
        Head hyperparameters, matching LorentzParT's classification head.
    encoder_kwargs : dict, optional
        Forwarded to LorentzParTEncoder.__init__ (num_layers, pair_embed_dims,
        ragged_pair_embed, ...). Must match the pretrained encoder's architecture.
        Note: at eval the encoder's BatchNorm uses frozen running stats and padded
        keys are masked in attention, so ``ragged_pair_embed`` does not change the
        output at valid positions — but pass it for a faithful reconstruction.
    """

    def __init__(
        self,
        encoder_weights: Optional[str] = None,
        embed_dim: int = 128,
        num_classes: int = 10,
        num_heads: int = 8,
        num_cls_layers: int = 2,
        hidden_dim: int = 256,
        num_mlp_layers: int = 0,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        kw = encoder_kwargs or {}
        self.processor = ParticleProcessor(to_multivector=True)
        self.encoder = LorentzParTEncoder(embed_dim=embed_dim, num_heads=num_heads, **kw)

        if encoder_weights is not None:
            state_dict = torch.load(encoder_weights, map_location='cpu', weights_only=True)
            filtered = {
                k[len('encoder.'):]: v
                for k, v in state_dict.items()
                if k.startswith('encoder.')
            }
            self.encoder.load_state_dict(filtered, strict=False)

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Trainable class-attention head (identical to LorentzParT's classification path)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)
        self.decoder = nn.ModuleList([
            ClassAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.0,  # matches LorentzParT — no dropout in class-attention blocks
                expansion_factor=expansion_factor,
            ) for _ in range(num_cls_layers)
        ])
        self.layernorm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(
            num_classes=num_classes,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            dropout=dropout,
        )

    def head_parameters(self):
        """Trainable head params (everything but the frozen encoder) — for the optimizer."""
        return [p for n, p in self.named_parameters() if not n.startswith('encoder.')]

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

        x_cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, embed_dim)
        for layer in self.decoder:
            x_cls = layer(embeddings, x_cls, padding_mask)

        x_cls = self.layernorm(x_cls).squeeze(1)  # (B, embed_dim)
        return self.classifier(x_cls)
