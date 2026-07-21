from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["t"] = state.get("t") + 1
                t = state["t"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad.data

                p.data.mul_(1 - lr * weight_decay) 
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1-beta1)

                exp_avg_sq.mul_(beta2) 
                exp_avg_sq.add_(grad**2, alpha=1-beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                adjusted_lr = lr * math.sqrt(bias_correction2) / bias_correction1

                denominator = torch.sqrt(exp_avg_sq) + eps

                update = exp_avg / denominator
                p.data.add_(update, alpha=-adjusted_lr)
        return loss

