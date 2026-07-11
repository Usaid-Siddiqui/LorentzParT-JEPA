from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine import Trainer
    

class BaseCallback:
    def on_train_begin(self, trainer: 'Trainer', logs: Optional[Dict[str, float]] = None) -> None:
        pass
    def on_train_end(self, trainer: 'Trainer', logs: Optional[Dict[str, float]] = None) -> None:
        pass
    def on_epoch_begin(self, epoch: int, trainer: 'Trainer', logs: Optional[Dict[str, float]] = None) -> None:
        pass
    def on_epoch_end(self, epoch: int, trainer: 'Trainer', logs: Optional[Dict[str, float]] = None) -> None:
        pass
    def on_batch_begin(self, batch: int, trainer: 'Trainer', logs: Optional[Dict[str, float]] = None) -> None:
        pass
    def on_batch_end(self, batch: int, trainer: 'Trainer', logs: Optional[Dict[str, float]] = None) -> None:
        pass


class EarlyStopping(BaseCallback):
    def __init__(
        self,
        monitor: str = 'val_loss',
        mode: str = 'min',
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ) -> None:
        self.monitor = monitor
        self.mode = mode  # 'min' (default, for loss) or 'max' (for accuracy)
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights = None

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return current < self.best_score - self.min_delta
        elif self.mode == 'max':
            return current > self.best_score + self.min_delta
        else:
            raise ValueError(f"Mode '{self.mode}' not supported.")

    def on_epoch_end(
        self,
        epoch: int,
        trainer: 'Trainer',
        logs: Optional[Dict[str, Any]] = None
    ) -> None:
        if logs is None:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_improvement(current):
            self.best_score = current
            self.wait = 0
            if self.restore_best_weights and trainer is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
        else:
            self.wait += 1

        if self.wait > self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None and trainer is not None:
                trainer.model.load_state_dict(self.best_weights)

            print(f"Early stopping at epoch {epoch + 1}. Best {self.monitor}: {self.best_score:.4f}")


class MaskCurriculum(BaseCallback):
    """Easy->hard masking curriculum: ramp the TRAIN dataset's num_mask over epochs.

    `schedule` is a list of [epoch, num_mask] breakpoints; at the start of each epoch
    num_mask is set to the value of the latest breakpoint whose epoch <= the current one
    (e.g. [[0, 1], [5, 2], [10, 4], [15, 8]] ramps K 1->2->4->8). Only the TRAIN loader's
    dataset is changed; validation stays at the config's num_mask so the val metric is a
    fixed-difficulty reference. Non-persistent DataLoader workers are re-forked each epoch,
    so they pick up the new value (set `persistent_workers=False`, the default)."""

    def __init__(self, schedule):
        self.schedule = sorted((int(e), int(k)) for e, k in schedule)

    def _num_mask_for(self, epoch: int) -> int:
        k = self.schedule[0][1]
        for e, kk in self.schedule:
            if epoch >= e:
                k = kk
        return k

    def on_epoch_begin(self, epoch: int, trainer: 'Trainer',
                       logs: Optional[Dict[str, float]] = None) -> None:
        k = self._num_mask_for(epoch)
        ds = getattr(getattr(trainer, 'train_loader', None), 'dataset', None)
        if ds is not None and hasattr(ds, 'num_mask') and ds.num_mask != k:
            ds.num_mask = k
            if getattr(trainer, 'rank', 0) == 0:
                print(f"[curriculum] epoch {epoch + 1}: num_mask -> {k}")


CALLBACK_REGISTRY = {
    'early_stopping': EarlyStopping,
    'mask_curriculum': MaskCurriculum,
    # Add more callbacks here as needed
}