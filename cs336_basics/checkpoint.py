import torch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model state": model.state_dict(),
        "optimizer state": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model_state = checkpoint["model state"]
    model.load_state_dict(model_state)
    optimizer_state = checkpoint["optimizer state"]
    optimizer.load_state_dict(optimizer_state)
    iteration = checkpoint["iteration"]
    return iteration