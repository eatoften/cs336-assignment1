import torch


def gradient_clipping(parameters, max_l2_norm):
    gradients = []
    for p in parameters:
        if p.grad is None:
            continue
        gradients.append(p.grad)
    total_squared_norm = 0
    for gradient in gradients:
        squared_contribution = torch.sum(gradient**2)
        total_squared_norm += squared_contribution
    if not gradients:
        return 
    total_norm = torch.sqrt(total_squared_norm)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for gradient in gradients:
            gradient.mul_(scale)
    return 
