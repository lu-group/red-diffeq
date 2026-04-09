import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional


def v_normalize(v):
    """Normalize velocity values to [-1, 1] range."""
    return (v - 1500) / 3000 * 2 - 1


def v_denormalize(v_norm):
    """Denormalize velocity values from [-1, 1] to physical range."""
    return (v_norm + 1) / 2 * 3000 + 1500


def s_normalize_none(s):
    """No normalization for seismic data."""
    return s


def s_normalize(s):
    """Normalize seismic data to [-1, 1] range."""
    return (s + 20) / 80 * 2 - 1


def s_denormalize(s_norm):
    """Denormalize seismic data from [-1, 1]."""
    return (s_norm + 1) / 2 * 80 - 20


def add_noise_to_seismic(y: torch.Tensor, std: float, noise_type: str = 'gaussian',
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Add noise to seismic data.

    Args:
        y: Seismic data tensor
        std: Standard deviation (Gaussian) or scale (Laplace)
        noise_type: 'gaussian' or 'laplace'
        generator: Optional torch.Generator for reproducibility

    Returns:
        Noisy seismic data
    """
    assert std >= 0, 'The standard deviation/scale of the noise must be greater than 0'
    assert noise_type in ['gaussian', 'laplace'], f'Unknown noise type: {noise_type}'

    if std == 0:
        return y

    device = y.device

    if noise_type == 'gaussian':
        noise = torch.randn(y.shape, generator=generator, device=device, dtype=y.dtype) * std
    elif noise_type == 'laplace':
        # Laplace distribution using inverse transform sampling
        # X = -b * sign(U) * log(1 - 2|U|) where U ~ Uniform(-0.5, 0.5)
        u = torch.rand(y.shape, generator=generator, device=device, dtype=y.dtype) - 0.5
        noise = -std * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))

    return y + noise


def prepare_initial_model(v_true: torch.Tensor, initial_type: str = None, sigma: float = None, linear_coeff: float = 1.0) -> torch.Tensor:
    """Prepare initial velocity model.

    Note: Uses scipy for Gaussian smoothing to ensure exact numerical consistency.
    This is only called once at initialization, so CPU/GPU transfer cost is negligible.

    Args:
        v_true: True velocity model
        initial_type: Type of initialization ('smoothed', 'homogeneous', 'linear')
        sigma: Smoothing parameter for 'smoothed' type
        linear_coeff: Coefficient for 'linear' type

    Returns:
        Initial velocity model (on same device as input)
    """
    assert initial_type in ['smoothed', 'homogeneous', 'linear'], \
        "please choose from 'smoothed', 'homogeneous', and 'linear'"

    device = v_true.device
    v = v_true.clone()

    v_np = v.cpu().numpy()
    v_np = v_normalize(v_np)

    if initial_type == 'smoothed':
        v_blurred = gaussian_filter(v_np, sigma=sigma)

    elif initial_type == 'homogeneous':
        min_top_row = np.min(v_np[0, 0, 0, :])
        v_blurred = np.full_like(v_np, min_top_row)

    elif initial_type == 'linear':
        v_min = np.min(v_np)
        v_max = np.max(v_np)
        height = v_np.shape[2]
        depth_gradient = np.linspace(v_min, v_max, height)
        depth_gradient = depth_gradient.reshape(-1, 1)
        v_blurred = np.tile(depth_gradient, (1, v_np.shape[3]))
        v_blurred = v_blurred.reshape(1, 1, height, -1)

    v_blurred = torch.tensor(v_blurred, dtype=torch.float32, device=device)

    return v_blurred


def missing_trace(y: torch.Tensor, num_missing: int, return_mask: bool = True,
                  generator: Optional[torch.Generator] = None):
    """Zero out random traces in seismic data and optionally return a mask.

    Missing trace indices are shared across all sources for each batch item.

    Args:
        y: Seismic data tensor of shape (batch, sources, time, traces)
        num_missing: Number of traces to zero out
        return_mask: If True, return (y_missing, mask). If False, return only y_missing (backward compatible)
        generator: Optional torch.Generator for reproducibility

    Returns:
        If return_mask=True: (y_missing, mask) where mask is 1 for observed traces, 0 for missing
        If return_mask=False: y_missing only (backward compatible)
    """
    assert num_missing >= 0, 'The number of missing traces must be >= 0'

    device = y.device
    batch_size, num_sources, time_samples, num_traces = y.shape

    mask = torch.ones_like(y, device=device)

    if num_missing == 0:
        if return_mask:
            return y, mask
        else:
            return y

    y_missing = y.clone()

    for b in range(batch_size):

        missing_indices = torch.randperm(num_traces, generator=generator, device=device)[:num_missing]
        y_missing[b, :, :, missing_indices] = 0
        mask[b, :, :, missing_indices] = 0

    if return_mask:
        return y_missing, mask
    else:
        return y_missing
