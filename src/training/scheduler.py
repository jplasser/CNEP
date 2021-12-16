import numpy as np

def assign_learning_rate(optimizer, new_lr):
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps, start_step=0, booster=0., elongation=1.):
    """
        start_step first step when running a resumed training
        booster adjusts learning rate curvature after warmup, range from 0.0 (no boost) to 1.0 (max boost)
        elongation static lr in the end for a proportion of elongation, 1. = no elongation, 1.2 20%
    """
    booster = np.max([2, 1 / (1e-11 + booster / 2)])
    steps = np.floor(steps / elongation)

    def _lr_adjuster(step):
        if (step - start_step) < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, (step - start_step))
        elif step <= steps:
            e = (step - start_step) - warmup_length
            es = (steps - start_step) - warmup_length
            lr = 0.5 * ((1 + np.cos(np.pi * e / es)) * base_lr) + (1 + np.cos(np.pi * (1 - e / es))) * base_lr / booster
        else:
            lr = 2 * base_lr / booster
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster