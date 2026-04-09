import torch
import torch.nn.functional as F

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))

def diffusion_pad(x: torch.Tensor) -> torch.Tensor:
    """Pad tensor for diffusion model input."""
    return F.pad(x, (1, 1, 1, 1), mode='constant', value=0)

def diffusion_crop(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, 1:-1, 1:-1]
