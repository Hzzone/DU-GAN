import torch
import torch.nn.functional as F
from kornia.filters import get_gaussian_kernel2d, filter2D


def compute_ssim(img1, img2, window_size=11, reduction: str = "mean", max_val: float = 1.0, full: bool = False):
    window: torch.Tensor = get_gaussian_kernel2d(
        (window_size, window_size), (1.5, 1.5))
    window = window.requires_grad_(False)
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2
    tmp_kernel: torch.Tensor = window.to(img1)
    tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)
    # compute local mean per channel
    mu1: torch.Tensor = filter2D(img1, tmp_kernel)
    mu2: torch.Tensor = filter2D(img2, tmp_kernel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # compute local sigma per channel
    sigma1_sq = filter2D(img1 * img1, tmp_kernel) - mu1_sq
    sigma2_sq = filter2D(img2 * img2, tmp_kernel) - mu2_sq
    sigma12 = filter2D(img1 * img2, tmp_kernel) - mu1_mu2

    ssim_map = ((2. * mu1_mu2 + C1) * (2. * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = ssim_map
    if reduction != 'none':
        ssim_score = torch.clamp(ssim_score, min=0, max=1)
        if reduction == "mean":
            ssim_score = torch.mean(ssim_score)
        elif reduction == "sum":
            ssim_score = torch.sum(ssim_score)
    if full:
        cs = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        return ssim_score, cs
    return ssim_score


def compute_psnr(input: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    mse_val = F.mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


def compute_rmse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(input, target))
