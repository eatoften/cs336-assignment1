import torch
import math


def lr_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1+math.cos(math.pi * progress))
        lr = min_learning_rate + cosine_factor * (max_learning_rate-min_learning_rate)
    else:
        lr = min_learning_rate
    return lr