import numpy as np

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps, start_step=0):
    def _lr_adjuster(step):
        if (step - start_step) < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, (step - start_step))
        else:
            e = (step - start_step) - warmup_length
            es = (steps - start_step) - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster