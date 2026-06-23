"""
JEPATrainer — training loop for ParticleJEPA pretraining.

Mirrors MaskedModelTrainer's interface (same batch format, same checkpointing)
but adds:
  - EMA target encoder update after every optimiser step
  - Linear momentum schedule (ema_start → ema_end over training)
  - Saves context_encoder weights with the "encoder." prefix so they load
    directly into LorentzParT without any changes to the fine-tuning scripts.
"""

import os
import time
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.distributed import all_gather, all_gather_object

from .trainer import Trainer
from ..utils import cleanup_ddp
from ..utils.data import JetClassDistributedSampler


class JEPATrainer(Trainer):
    """
    Trainer for ParticleJEPA self-supervised pretraining.

    Expects batches of shape (X, y, mask_idx), the same format as
    MaskedModelTrainer, so the same JetClassDataset with mask_mode='random'
    can be used for both MAE and JEPA pretraining.

    Parameters
    ----------
    ema_momentum_start : float
        Initial EMA momentum for the target encoder (default 0.996).
    ema_momentum_end : float
        Final EMA momentum, reached linearly at the last epoch (default 1.0).
    All other parameters are passed to the base Trainer.
    """

    def __init__(
        self,
        *args,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ema_momentum_start = ema_momentum_start
        self.ema_momentum_end = ema_momentum_end
        self.history = {
            'epoch': [],
            'embedding_loss': [],
            'val_loss': [],
            'epoch_time_s': [],
            'elapsed_total_s': [],
        }
        self._train_start_time: Optional[float] = None
        self.best_epoch: int = 0

    def _get_momentum(self, step: int, total_steps: int) -> float:
        """Linearly ramp EMA momentum from start to end over all training steps."""
        progress = min(step / max(total_steps, 1), 1.0)
        return self.ema_momentum_start + progress * (self.ema_momentum_end - self.ema_momentum_start)

    def _unwrap(self):
        """Return the underlying model, unwrapping DDP if needed."""
        return self.model.module if self._is_distributed else self.model

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            self._train_start_time = time.monotonic()
            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)

            if self.progress_bar and self.rank == 0:
                global_bar = tqdm(
                    total=total_steps,
                    initial=start_step,
                    desc="JEPA Pretraining",
                    dynamic_ncols=True,
                )
            else:
                class _NoOpBar:
                    def set_postfix(self, *args, **kwargs): pass
                    def update(self, *args, **kwargs): pass
                global_bar = _NoOpBar()

            global_step = start_step

            for epoch in range(self.start_epoch, self.num_epochs):
                epoch_start = time.monotonic()

                if self._is_distributed and isinstance(
                    self.train_loader.batch_sampler, JetClassDistributedSampler
                ):
                    self.train_loader.batch_sampler.set_epoch(epoch)

                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                # ------ Training phase ------
                self.model.train()
                running_loss_sum = 0.0
                running_count = 0

                for batch_idx, (X, y, mask_idx) in enumerate(self.train_loader):
                    global_step += 1
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    mask_idx = mask_idx.to(self.device).long().squeeze(-1)

                    self.optimizer.zero_grad()
                    pred, target = self._unwrap()(X, mask_idx)
                    loss, components = self.criterion(pred, target)
                    loss.backward()
                    self.optimizer.step()

                    # EMA update with linearly increasing momentum
                    momentum = self._get_momentum(global_step, total_steps)
                    self._unwrap().update_target_encoder(momentum)

                    bsz = X.size(0)
                    running_loss_sum += float(loss.item()) * bsz
                    running_count += bsz
                    avg_loss = running_loss_sum / running_count

                    if self.rank == 0:
                        if step % self.logging_steps == 0 or step == total_steps:
                            tqdm.write(
                                f"step: {step}/{total_steps} | "
                                f"embedding_loss: {avg_loss:.6f} | "
                                f"ema_momentum: {momentum:.5f}"
                            )
                        global_bar.set_postfix({
                            "epoch": f"{epoch + 1}/{self.num_epochs}",
                            "loss": f"{avg_loss:.6f}",
                            "ema_m": f"{momentum:.4f}",
                        })

                    global_bar.update(1)

                # ------ Validation phase ------
                self.model.eval()
                val_loss_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for X_val, y_val, mask_idx_val in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        mask_idx_val = mask_idx_val.to(self.device).long().squeeze(-1)

                        pred_val, target_val = self._unwrap()(X_val, mask_idx_val)
                        loss_val, _ = self.criterion(pred_val, target_val)
                        bsz = X_val.size(0)
                        val_loss_sum += float(loss_val.item()) * bsz
                        val_count += bsz

                # Gather across processes if distributed
                if self._is_distributed:
                    pack = torch.tensor(
                        [val_loss_sum, float(val_count)],
                        dtype=torch.float64,
                        device=self.device,
                    )
                    packs = [torch.zeros_like(pack) for _ in range(self.world_size)]
                    all_gather(packs, pack)
                    total_val_loss_sum = sum(p[0].item() for p in packs)
                    total_val_count = int(sum(p[1].item() for p in packs))
                else:
                    total_val_loss_sum = val_loss_sum
                    total_val_count = val_count

                val_loss = total_val_loss_sum / max(total_val_count, 1)

                if self.rank == 0:
                    tqdm.write(
                        f"epoch: {epoch + 1}/{self.num_epochs} | "
                        f"train_loss: {avg_loss:.6f} | "
                        f"val_loss: {val_loss:.6f}"
                    )

                if self.scheduler:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']

                epoch_time = time.monotonic() - epoch_start
                elapsed_total = time.monotonic() - self._train_start_time

                # Save best model (context_encoder weights, compatible with LorentzParT loader)
                if self.best_model_path and self.rank == 0 and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self._save_context_encoder_weights(self.best_model_path)

                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['embedding_loss'].append(avg_loss)
                self.history['val_loss'].append(val_loss)
                self.history['epoch_time_s'].append(epoch_time)
                self.history['elapsed_total_s'].append(elapsed_total)

                self.save_checkpoint(epoch)

                logs = {
                    'epoch': epoch + 1,
                    'embedding_loss': avg_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    'ema_momentum': momentum,
                    'epoch_time_s': epoch_time,
                    'elapsed_total_s': elapsed_total,
                    'best_epoch': self.best_epoch,
                }
                self.log_csv(logs)

                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # JEPA: the val embedding loss is a noisy, poor proxy for representation
            # quality, and the representation keeps consolidating as the EMA target
            # anneals to 1.0. Save the FINAL context encoder (full schedule, stabilized
            # target) — overrides the best-val checkpoint, which is from an early,
            # under-annealed epoch. Pair with early stopping disabled in the config.
            if self.best_model_path and self.rank == 0:
                print(f"  Saving FINAL context encoder (epoch {epoch + 1}, EMA annealed)")
                self._save_context_encoder_weights(self.best_model_path)

            for cb in self.callbacks:
                cb.on_train_end(trainer=self)

        except KeyboardInterrupt:
            if self.rank == 0:
                print(f"\nJEPA pretraining interrupted at epoch {epoch + 1}.")
            cleanup_ddp()

        return self.history, self.model

    def _save_context_encoder_weights(self, path: str):
        """
        Save the context encoder's weights with the "encoder." prefix.

        LorentzParT.__init__ loads pretrained weights by filtering for keys
        that start with "encoder." and stripping that prefix. By saving with
        this convention we remain fully compatible with the existing fine-tuning
        scripts without any modifications.
        """
        model = self._unwrap()
        state_dict = {
            f"encoder.{k}": v
            for k, v in model.context_encoder.state_dict().items()
        }
        torch.save(state_dict, path)
        if self.rank == 0:
            print(f"  Saved context encoder weights → {path}")

    def save_checkpoint(self, epoch: int):
        """Override to also store JEPA-specific fields in the checkpoint."""
        if not self.checkpoint_path or self.rank != 0:
            return
        model_state = self._unwrap().state_dict()
        checkpoint = {
            'run_name': self.run_name,
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'ema_momentum_start': self.ema_momentum_start,
            'ema_momentum_end': self.ema_momentum_end,
        }
        torch.save(checkpoint, self.checkpoint_path)

    @torch.no_grad()
    def evaluate(self, plot=None):
        """Evaluate embedding loss on the test set."""
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")

        self.model.eval()
        loss_sum = 0.0
        count = 0

        for X_test, y_test, mask_idx_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            mask_idx_test = mask_idx_test.to(self.device).long().squeeze(-1)

            pred, target = self._unwrap()(X_test, mask_idx_test)
            loss, _ = self.criterion(pred, target)
            bsz = X_test.size(0)
            loss_sum += float(loss.item()) * bsz
            count += bsz

        test_loss = loss_sum / max(count, 1)
        if self.rank == 0:
            print(f"JEPA test embedding_loss: {test_loss:.6f}")

        return test_loss
