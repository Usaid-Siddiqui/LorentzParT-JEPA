from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class BaseModelConfig:
    num_classes: int = 10
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 8
    num_cls_layers: int = 2
    num_mlp_layers: int = 0
    hidden_dim: int = 256
    dropout: float = 0.1
    max_num_particles: int = 128
    num_particle_features: int = 4
    expansion_factor: int = 4
    mask: bool = False
    weights: Optional[str] = None
    inference: bool = False


@dataclass
class ParticleTransformerConfig(BaseModelConfig):
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])

    @classmethod
    def from_dict(cls, d: Dict):
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class LGATrConfig(BaseModelConfig):
    hidden_mv_channels: int = 8
    in_s_channels: int = None
    out_s_channels: int = None
    hidden_s_channels: int = 16
    attention: Optional[Dict] = None
    mlp: Optional[Dict] = None
    reinsert_mv_channels: Optional[Tuple[int]] = None
    reinsert_s_channels: Optional[Tuple[int]] = None

    @classmethod
    def from_dict(cls, d: Dict):
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class LorentzParTConfig(BaseModelConfig):
    hidden_mv_channels: int = 8
    in_s_channels: int = None
    out_s_channels: int = None
    hidden_s_channels: int = 16
    attention: Optional[Dict] = None
    mlp: Optional[Dict] = None
    reinsert_mv_channels: Optional[Tuple[int]] = None
    reinsert_s_channels: Optional[Tuple[int]] = None
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])

    @classmethod
    def from_dict(cls, d: Dict):
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class JEPAConfig(BaseModelConfig):
    """
    Configuration for ParticleJEPA pretraining.

    Supports both flat dicts and nested 'predictor'/'jepa' sub-dicts
    (as produced by the YAML configs).
    """
    pair_embed_dims: List[int] = field(default_factory=lambda: [64, 64, 64])
    predictor_dim: int = 64
    predictor_heads: int = 4
    predictor_layers: int = 4
    predictor_dropout: float = 0.1
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 1.0
    use_attention_gate: bool = True
    num_mask: int = 1

    @classmethod
    def from_dict(cls, d: Dict):
        # Flatten nested 'predictor' and 'jepa' sub-dicts if present
        flat = dict(d)
        if 'predictor' in flat:
            pred = flat.pop('predictor')
            flat.setdefault('predictor_dim', pred.get('predictor_dim', 64))
            flat.setdefault('predictor_heads', pred.get('num_heads', 4))
            flat.setdefault('predictor_layers', pred.get('num_layers', 4))
            flat.setdefault('predictor_dropout', pred.get('dropout', 0.1))
        if 'jepa' in flat:
            jepa = flat.pop('jepa')
            flat.setdefault('ema_momentum_start', jepa.get('ema_momentum_start', 0.996))
            flat.setdefault('ema_momentum_end', jepa.get('ema_momentum_end', 1.0))
            flat.setdefault('use_attention_gate', jepa.get('use_attention_gate', True))
            flat.setdefault('num_mask', jepa.get('num_mask', 1))
        # Only pass fields that JEPAConfig actually declares
        valid_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in flat.items() if k in valid_fields})
