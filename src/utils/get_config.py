import inspect
from typing import List, Dict

from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .callbacks import BaseCallback


def _cfg_kwargs(cfg: Dict) -> Dict:
    """Constructor kwargs from a config block, accepting BOTH spellings.

    Every YAML in this repo writes params flat (``name: radam`` / ``lr: 0.0005``), but the getters
    only read a nested ``kwargs:`` block — so lr, betas, Lookahead k/alpha, loss params and
    callback params were all silently dropped and every run trained on library defaults
    (Phase 8). The scheduler getter alone had been fixed; this generalises that fix.
    Nested ``kwargs`` wins on a clash.
    """
    flat = {k: v for k, v in cfg.items() if k not in ('name', 'kwargs')}
    return {**flat, **(cfg.get('kwargs') or {})}


def get_loss_from_config(loss_config: Dict, registry: Dict) -> _Loss:
    name = loss_config['name']
    kwargs = _cfg_kwargs(loss_config)

    if name not in registry:
        raise ValueError(f"Loss function '{name}' not found in registry.")
    
    criterion = registry[name]
    sig = inspect.signature(criterion.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }
    
    return criterion(**valid_args)


def get_optim_from_config(optim_config: Dict, registry: Dict, model: nn.Module) -> Optimizer:
    name = optim_config['name']
    kwargs = _cfg_kwargs(optim_config)

    if name not in registry:
        raise ValueError(f"Optimizer '{name}' not found in registry.")
    
    optimizer = registry[name]
    sig = inspect.signature(optimizer.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }

    # Some optimizers (e.g. RAdam) expect tuple inputs for betas; YAML stores them as lists
    if 'betas' in valid_args and isinstance(valid_args['betas'], (list, tuple)):
        valid_args['betas'] = tuple(valid_args['betas'])

    return optimizer(model.parameters(), **valid_args)


def get_optim_wrapper_from_config(optim_wrapper_config: Dict, registry: Dict, optimizer: Optimizer) -> Optimizer:
    name = optim_wrapper_config['name']
    kwargs = _cfg_kwargs(optim_wrapper_config)

    if name not in registry:
        raise ValueError(f"Optimizer wrapper '{name}' not found in registry.")
    
    optim_wrapper = registry[name]
    sig = inspect.signature(optim_wrapper.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }

    return optim_wrapper(optimizer, **valid_args)


def get_scheduler_from_config(scheduler_config: Dict, registry: Dict, optimizer: Optimizer) -> _LRScheduler:
    name = scheduler_config['name']
    # Accept kwargs either nested under 'kwargs' or flat at the top level
    kwargs = _cfg_kwargs(scheduler_config)

    if name not in registry:
        raise ValueError(f"Scheduler '{name}' not found in registry.")

    scheduler = registry[name]
    sig = inspect.signature(scheduler.__init__)
    valid_args = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters and k != 'self'
    }

    return scheduler(optimizer, **valid_args)


def get_callbacks_from_config(callbacks_config: List[Dict], registry: Dict) -> List[BaseCallback]:
    callbacks = []

    for cb_cfg in callbacks_config:
        name = cb_cfg['name']
        kwargs = _cfg_kwargs(cb_cfg)

        if name not in registry:
            raise ValueError(f"Callback '{name}' not found in registry.")
        
        callback_cls = registry[name]
        sig = inspect.signature(callback_cls.__init__)
        valid_args = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters and k != 'self'
        }

        callbacks.append(callback_cls(**valid_args))
    
    return callbacks