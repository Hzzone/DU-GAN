import torch
import torch.nn.functional as F


def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)


def standard_gan(inputs, targets):
    if isinstance(targets, float):
        targets = torch.ones_like(inputs) * targets
    return F.binary_cross_entropy(inputs, targets)
