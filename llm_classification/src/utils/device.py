import torch


def resolve_device(device=None):
    if device is not None:
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
