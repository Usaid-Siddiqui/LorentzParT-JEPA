"""
Phase 4 transfer protocols — name → model builder. One place that knows how each
protocol wires the (frozen or trainable) encoder to a k-class head. All three
reconstruct the encoder ragged, matching the 1M pretrains.

  finetune : LorentzParT, encoder-initialized (or random) + trainable — the ceiling
  linear   : LinearProbeModel        — frozen encoder + mean-pool + linear (floor)
  pool     : AttentivePoolProbeModel  — frozen encoder + 1-query attention pool (I-JEPA)

``encoder_weights=None`` gives the scratch column (random init): a random-*trainable*
encoder for finetune, a random-*frozen* encoder (random-feature control) for the probes.
"""

from src.models import LorentzParT, LinearProbeModel, AttentivePoolProbeModel

PROTOCOLS = ('finetune', 'linear', 'pool')

# encoder architecture shared by all protocols (matches the 1M ragged pretrains)
_ENCODER_KWARGS = {
    'num_layers': 8,
    'dropout': 0.1,
    'expansion_factor': 4,
    'pair_embed_dims': [64, 64, 64],
    'ragged_pair_embed': True,
}


def is_frozen(protocol: str) -> bool:
    return protocol in ('linear', 'pool')


def build_model(protocol: str, num_classes: int, encoder_weights=None, num_extra_features: int = 0):
    """Return a model for (protocol, k). encoder_weights=None → scratch/random.
    num_extra_features > 0 (finetune only) adds extra per-particle scalar inputs
    (e.g. 4 track-displacement features) beyond the 4-vector — the input-ceiling test."""
    if protocol == 'finetune':
        return LorentzParT(
            num_classes=num_classes,
            weights=encoder_weights,        # LorentzParT filters encoder.* internally; None → random
            ragged_pair_embed=True,
            mask=False,
            num_extra_features=num_extra_features,
        )
    if protocol == 'linear':
        return LinearProbeModel(
            encoder_weights=encoder_weights,
            num_classes=num_classes,
            encoder_kwargs=_ENCODER_KWARGS,
        )
    if protocol == 'pool':
        return AttentivePoolProbeModel(
            encoder_weights=encoder_weights,
            num_classes=num_classes,
            encoder_kwargs=_ENCODER_KWARGS,
        )
    raise ValueError(f"unknown protocol {protocol!r} (expected one of {PROTOCOLS})")
