import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def cutmix(
    x: torch.tensor,
    y: torch.tensor = None,
    cutmix_prob: int = 0.1,
    beta: int = 0.3,
    num_classes: int = 60,
) -> torch.tensor:
    if y == None:
        y = torch.zeros((x.shape[0])).to(x.device)
    if np.random.rand() > cutmix_prob:
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=num_classes).float()
        return x, y
    N, _, H, W = x.shape
    indices = torch.randperm(N).to(x.device)
    label = torch.zeros((N, num_classes)).to(x.device)
    x1 = x[indices, :, :, :]
    y1 = y.clone()[indices]
    lam = np.random.beta(beta, beta)
    rate = np.sqrt(1 - lam)
    cut_x, cut_y = int((H * rate) // 2), int((W * rate) // 2)
    if cut_x == H // 2 or cut_y == W // 2:
        if y.ndim == 1:
            y = F.one_hot(y, num_classes=num_classes).float()
        return x, y
    cx, cy = int(np.random.randint(cut_x, H - cut_x)), int(np.random.randint(cut_y, W - cut_x))
    bx1, bx2 = cx - cut_x, cx + cut_x
    by1, by2 = cy - cut_y, cy + cut_y
    x[:, :, bx1:bx2, by1:by2] = x1[:, :, bx1:bx2, by1:by2].clone()
    label[torch.arange(N), y] = lam
    label[torch.arange(N), y1] = 1 - lam
    return x, label
