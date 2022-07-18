import numpy as np
import torch


@torch.no_grad()
def mixup(
    x: torch.tensor, y: torch.tensor, cutmix_prob: int = 0.5, beta: int = 0.1, num_classes: int = 60
) -> torch.tensor:
    if np.random.rand() > cutmix_prob:
        return x, y
    N, _, H, W = x.shape
    indices = torch.randperm(N, device=torch.device("cuda"))
    label = torch.zeros((N, num_classes), device=torch.device("cuda"))
    x1 = x[indices, :, :, :].clone()
    y1 = y.clone()[indices]
    lam = np.random.beta(beta, beta)

    x = lam * x + (1 - lam) * x1
    label[torch.arange(N), y] = lam
    label[torch.arange(N), y1] = 1 - lam
    return x, label
