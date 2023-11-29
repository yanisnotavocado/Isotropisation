import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def interpolate1d(img, size, dim, **kwargs):
    img = img.transpose(-1, dim)
    shape = [*img.shape[:-1], size]
    img = img.view(-1, img.size(-2), img.size(-1))
    img = F.interpolate(img, size=size, mode="linear", **kwargs)
    img = img.view(*shape)
    img = img.transpose(-1, dim)
    return img

def add_noise(img, std): return (img + torch.randn(*img.shape).to(img) * std) / ((1 + std ** 2) ** 0.5)

def kld_annealing_beta(step, pediod: int, eps: float = 1e-3):
    return np.maximum(np.cos(((step % pediod) / pediod - 1) * np.pi) * 0.5 + 0.5, eps)

def replace_upconv(model: nn.Module, dim: int = 2, mode: str = "nearest"):
    for name, module in model.named_children():
        if isinstance(module, getattr(nn, f"ConvTranspose{dim}d")):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            conv = getattr(nn, f"Conv{dim}d")(in_channels, out_channels, kernel_size, 1, kernel_size//2)
            setattr(model, name, conv)
            if stride != 1:
                up = nn.Upsample(scale_factor=stride, mode=mode)
                setattr(model, name, nn.Sequential(up, conv))
        else:
            replace_upconv(module, dim=dim, mode=mode)