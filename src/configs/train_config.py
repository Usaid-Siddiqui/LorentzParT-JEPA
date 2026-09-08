from dataclasses import dataclass, field, fields


@dataclass
class TrainConfig:
    batch_size: int = 64
    criterion: dict = field(default_factory=lambda: {'name': 'cross_entropy_loss', 'kwargs': {}})
    optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'kwargs': {'lr': 1e-4}})
    optimizer_wrapper: dict = None
    scheduler: dict = None
    callbacks: list = None
    num_epochs: int = 20
    start_epoch: int = 0
    logging_dir: str = 'logs'
    logging_steps: int = 500
    progress_bar: bool = True
    save_best: bool = True
    save_ckpt: bool = True
    device: str = None
    num_workers: int = 0
    pin_memory: bool = False
    # Phase 7 streaming / scale knobs (ignored by map-style datasets when unset)
    steps_per_epoch: int = None   # optimizer steps per epoch for IterableDataset (None = full pass)
    val_steps: int = None         # cap validation batches per epoch (None = full val set)
    amp: str = None               # 'bf16' | 'fp16' | None (mixed-precision autocast)

    @classmethod
    def from_dict(cls, d: dict):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})