from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    """
    PyTorch implementation of the Lookahead optimizer wrapper.

    Parameters
    ----------
    optimizer: Optimizer
        The inner optimizer.
    la_steps: int
        Number of lookahead steps.
    la_alpha: float
        Linear interpolation factor. 1.0 recovers the inner optimizer.
    pullback_momentum: str
        Change to inner optimizer momentum on interpolation update.

    .. References::
        Michael Zhang, James Lucas, Jimmy Ba, and Geoffrey E Hinton.
        [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610).
        *Advances in Neural Information Processing Systems*, 32, 2019.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        la_steps: int = 5,
        la_alpha: float = 0.8,
        pullback_momentum: str = 'none'
    ):
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        """Inner optimizer state PLUS Lookahead's own slow weights and step counter.

        Phase 8: previously only the inner optimizer was saved, so a resumed run silently
        restarted the Lookahead cycle and discarded the slow weights.
        """
        inner = self.optimizer.state_dict()
        cached = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                cached.append(self.state[p].get('cached_params'))
        return {'inner': inner, 'la_state': {'_la_step': self._la_step,
                                             'la_alpha': self.la_alpha,
                                             'cached_params': cached}}

    def load_state_dict(self, state_dict):
        if 'la_state' not in state_dict:          # legacy checkpoint: inner state only
            self.optimizer.load_state_dict(state_dict)
            return
        self.optimizer.load_state_dict(state_dict['inner'])
        la = state_dict['la_state']
        self._la_step = la['_la_step']
        self.la_alpha = la['la_alpha']
        cached = la['cached_params']
        i = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if i < len(cached) and cached[i] is not None:
                    self.state[p]['cached_params'] = cached[i].to(p.device)
                i += 1

    def _backup_and_load_cache(self):
        """
        Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])

                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """
        Performs a single Lookahead optimization step.

        Parameters
        ----------
            closure: Callable, optional
                A closure that reevaluates the model and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1
        if self._la_step >= self._total_la_steps:
            self._la_step = 0

            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"]
                        )
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss