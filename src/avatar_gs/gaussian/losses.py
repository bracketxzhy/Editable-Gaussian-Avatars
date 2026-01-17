import torch

def l2_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - gt) ** 2)
