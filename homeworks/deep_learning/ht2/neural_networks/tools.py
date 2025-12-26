import torch
import torch.nn.functional as F
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def median_filter(img: torch.Tensor, k: int = 3) -> torch.Tensor:
    # to (1, C, H, W)
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)  # (1, C, H, W)

    p = k // 2
    x = F.pad(img, (p, p, p, p), mode='reflect')

    windows = x.unfold(2, k, step=1).unfold(3, k, step=1)  # (B,C,H,W,k,k)
    windows = windows.contiguous().view(*windows.shape[:4], -1)  # (B,C,H,W,k*k)

    result = windows.median(dim=-1).values

    return result.squeeze(0)  # (C, H, W)
