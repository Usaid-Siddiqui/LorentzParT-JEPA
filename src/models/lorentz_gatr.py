"""
LorentzGATr — Full L-GATr model for jet classification and MAE pretraining.

Uses the external `lgatr` library's LGATr backbone directly, rather than the
LorentzParT hybrid (which wraps a ParT-style attention stack with a single
EquiLinear projection). LorentzGATr is a deeper Lorentz-equivariant model —
every transformer block operates natively in the geometric algebra (Cl(1,3))
multivector space.

Architecture (classification mode):
    ParticleProcessor → LGATrEncoder (full L-GATr backbone, N blocks)
    → Linear(16 → embed_dim) → CLS token → ClassAttentionBlock × num_cls_layers
    → Linear(embed_dim → num_classes)

Architecture (mask/MAE mode):
    ParticleProcessor → LGATrEncoder → Linear(N×16 → 4)
    Returns reconstructed (pT, η, φ, E) for the masked particle.

Used as a baseline in ablation studies to isolate the contribution of:
  - Lorentz equivariance at every layer (vs LorentzParT's single EquiLinear)
  - ParT-style pairwise interaction biases (vs L-GATr's attention)

References:
    Brehmer et al. "A Lorentz-Equivariant Transformer for All of the LHC."
    arXiv:2411.00446, 2024.

    Spinner et al. "Lorentz-Equivariant Geometric Algebra Transformers for
    High-Energy Physics." NeurIPS 2024. arXiv:2405.14806.

    Brehmer et al. "Geometric Algebra Transformer." NeurIPS 2023.
    arXiv:2305.18415.
"""

from typing import Tuple, Dict, Optional

import torch
from torch import nn, Tensor
from lgatr import LGATr
from lgatr.interface import extract_vector

from .classifier import ClassAttentionBlock, Classifier
from .processor import ParticleProcessor
from ..configs import LGATrConfig


class LGATrEncoder(nn.Module):
    """
    Thin wrapper around the external LGATr backbone.

    Handles the (B, N, 16) ↔ (B, N, 1, 16) reshape required by the lgatr API
    and returns a flat (B, N, 16) multivector sequence.

    Parameters
    ----------
    num_layers : int
        Number of L-GATr transformer blocks (default 8).
    hidden_mv_channels : int
        Hidden multivector channels inside each block (default 8).
    in_s_channels : int, optional
        Scalar input channels (None = no scalar stream).
    out_s_channels : int, optional
        Scalar output channels (None = no scalar stream).
    hidden_s_channels : int, optional
        Hidden scalar channels inside each block (default 16).
    attention : dict
        Keyword arguments forwarded to the L-GATr attention module.
    mlp : dict
        Keyword arguments forwarded to the L-GATr MLP module.
    reinsert_mv_channels : tuple of int, optional
        Channel indices to reinsert at each block (L-GATr skip connections).
    reinsert_s_channels : tuple of int, optional
        Scalar channel indices to reinsert at each block.
    dropout : float, optional
        Dropout probability (None = no dropout).
    """

    def __init__(
        self,
        num_layers: int = 8,
        hidden_mv_channels: int = 8,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        hidden_s_channels: Optional[int] = 16,
        attention: Dict = {},
        mlp: Dict = {},
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.encoder = LGATr(
            num_blocks=num_layers,
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=attention,
            mlp=mlp,
            reinsert_mv_channels=reinsert_mv_channels,
            reinsert_s_channels=reinsert_s_channels,
            dropout_prob=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, N, 16)
            Lorentz multivectors from ParticleProcessor.

        Returns
        -------
        Tensor, shape (B, N, 16)
            Encoded multivector sequence.
        """
        B, N, F = x.shape
        x = x.view(B, N, 1, F)     # (B, N, 1, 16) — lgatr API requires channel dim
        x, _ = self.encoder(x)     # (B, N, out_mv_channels=1, 16)
        return x.view(B, N, -1)    # (B, N, 16)


class LorentzGATr(nn.Module):
    """
    Full L-GATr model for jet classification and MAE pretraining.

    Parameters
    ----------
    config : LGATrConfig, optional
        Config object. Individual kwargs below override config values if provided.
    num_classes : int
        Number of output classes (default 10).
    embed_dim : int
        Embedding dimension for the classification head (default 128).
    num_heads : int
        Attention heads in ClassAttentionBlock (default 8).
    num_layers : int
        Number of L-GATr transformer blocks (default 8).
    num_cls_layers : int
        Number of ClassAttentionBlocks in the classification head (default 2).
    hidden_dim : int
        Hidden dimension in the Classifier MLP (default 256).
    hidden_mv_channels : int
        Hidden multivector channels in each L-GATr block (default 8).
    hidden_s_channels : int
        Hidden scalar channels in each L-GATr block (default 16).
    dropout : float, optional
        Dropout probability (None = disabled).
    expansion_factor : int
        FFN expansion factor in ClassAttentionBlock (default 4).
    max_num_particles : int
        Maximum particles per jet (default 128).
    mask : bool
        If True, runs in MAE reconstruction mode instead of classification.
    weights : str, optional
        Path to pretrained encoder checkpoint.
    inference : bool
        If True, applies Softmax to classification output.
    """

    def __init__(
        self,
        config: Optional[LGATrConfig] = None,
        num_classes: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_cls_layers: Optional[int] = None,
        num_mlp_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        hidden_s_channels: Optional[int] = None,
        attention: Optional[Dict] = None,
        mlp: Optional[Dict] = None,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        dropout: Optional[float] = None,
        expansion_factor: Optional[int] = None,
        max_num_particles: Optional[int] = None,
        num_particle_features: Optional[int] = None,
        mask: Optional[bool] = None,
        weights: Optional[str] = None,
        inference: Optional[bool] = False,
    ):
        super().__init__()

        # ── Resolve config vs kwargs ──────────────────────────────────────────
        def _r(kwarg, attr, default):
            """Return kwarg if given, else config attr if config exists, else default."""
            if kwarg is not None:
                return kwarg
            return getattr(config, attr, default) if config is not None else default

        self.num_classes         = _r(num_classes,         'num_classes',         10)
        self.embed_dim           = _r(embed_dim,           'embed_dim',           128)
        self.num_heads           = _r(num_heads,           'num_heads',           8)
        self.num_layers          = _r(num_layers,          'num_layers',          8)
        self.num_cls_layers      = _r(num_cls_layers,      'num_cls_layers',      2)
        self.num_mlp_layers      = _r(num_mlp_layers,      'num_mlp_layers',      0)
        self.hidden_dim          = _r(hidden_dim,          'hidden_dim',          256)
        self.hidden_mv_channels  = _r(hidden_mv_channels,  'hidden_mv_channels',  8)
        self.in_s_channels       = _r(in_s_channels,       'in_s_channels',       None)
        self.out_s_channels      = _r(out_s_channels,      'out_s_channels',      None)
        self.hidden_s_channels   = _r(hidden_s_channels,   'hidden_s_channels',   16)
        self.attention           = _r(attention,           'attention',           {})
        self.mlp                 = _r(mlp,                 'mlp',                 {})
        self.reinsert_mv_channels = _r(reinsert_mv_channels, 'reinsert_mv_channels', None)
        self.reinsert_s_channels  = _r(reinsert_s_channels,  'reinsert_s_channels',  None)
        self.dropout             = _r(dropout,             'dropout',             None)
        self.expansion_factor    = _r(expansion_factor,    'expansion_factor',    4)
        self.max_num_particles   = _r(max_num_particles,   'max_num_particles',   128)
        self.num_particle_features = _r(num_particle_features, 'num_particle_features', 4)
        self.mask                = _r(mask,                'mask',                False)
        self.weights             = _r(weights,             'weights',             None)
        self.inference           = _r(inference,           'inference',           False)

        # ── Submodules ────────────────────────────────────────────────────────
        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

        # Shared processor: (pT, η, φ, E) → 16-dim multivectors
        self.processor = ParticleProcessor(to_multivector=True)

        # Full L-GATr encoder backbone
        self.encoder = LGATrEncoder(
            num_layers=self.num_layers,
            hidden_mv_channels=self.hidden_mv_channels,
            in_s_channels=self.in_s_channels,
            out_s_channels=self.out_s_channels,
            hidden_s_channels=self.hidden_s_channels,
            attention=self.attention,
            mlp=self.mlp,
            reinsert_mv_channels=self.reinsert_mv_channels,
            reinsert_s_channels=self.reinsert_s_channels,
            dropout=self.dropout,
        )

        # MAE reconstruction head: flatten → predict raw features
        self.fc = nn.Linear(self.max_num_particles * 16, self.num_particle_features)

        # Classification head: project 16-dim mv → embed_dim, then class attention
        self.proj = nn.Linear(16, self.embed_dim)
        self.decoder = nn.ModuleList([
            ClassAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,
                expansion_factor=self.expansion_factor,
            )
            for _ in range(self.num_cls_layers)
        ])
        self.layernorm  = nn.LayerNorm(self.embed_dim)
        self.classifier = Classifier(
            num_classes=self.num_classes,
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_mlp_layers,
            dropout=self.dropout,
        )
        self.act = nn.Softmax(dim=1) if self.inference else nn.Identity()

        # ── Load pretrained weights ───────────────────────────────────────────
        if self.weights is not None:
            state_dict = torch.load(self.weights)
            filtered = {
                k[len("encoder."):]: v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(filtered, strict=False)

    def forward(self, x: Tensor, mask_idx: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, N, 4)
            Particle features [pT, η, φ, E], zero-padded.
        mask_idx : Tensor, shape (B,), optional
            Masked particle indices (MAE mode only).

        Returns
        -------
        Tensor
            Classification logits (B, num_classes) or reconstructed features (B, 4).
        """
        B, N, F = x.shape

        # ── Padding mask ──────────────────────────────────────────────────────
        padding_mask = (x[..., 3] == 0).float()   # (B, N)
        if mask_idx is not None:
            batch_idx = torch.arange(B, device=x.device)
            padding_mask[batch_idx, mask_idx.view(-1)] = 0.0   # keep masked pos active

        # ── Encode ───────────────────────────────────────────────────────────
        mv, _ = self.processor(x)    # (B, N, 16) — U not used by L-GATr
        x = self.encoder(mv)         # (B, N, 16)

        # ── Classification mode ───────────────────────────────────────────────
        if not self.mask:
            x_cls = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
            x = self.proj(x)                            # (B, N, embed_dim)

            for layer in self.decoder:
                x_cls = layer(x, x_cls, padding_mask)

            x_cls = self.layernorm(x_cls).squeeze(1)    # (B, embed_dim)
            return self.act(self.classifier(x_cls))     # (B, num_classes)

        # ── MAE reconstruction mode ───────────────────────────────────────────
        else:
            x = x.view(B, -1)    # (B, N * 16)
            return self.fc(x)    # (B, num_particle_features)
