from torch.optim import lr_scheduler


SCHEDULER_REGISTRY = {
    'exponential_lr': lr_scheduler.ExponentialLR,
    'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,
    'cosine_annealing_lr': lr_scheduler.CosineAnnealingLR,
    'step_lr': lr_scheduler.StepLR,
}