import torch
from typing import Tuple
from red_diffeq.utils.data_trans import v_normalize
from red_diffeq.utils.ssim import SSIM


class MetricsCalculator:
    """Calculate evaluation metrics for velocity models."""

    def __init__(self, ssim_loss: SSIM):
        self.ssim_loss = ssim_loss

    def calculate(self, mu: torch.Tensor, mu_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate per-sample MAE, RMSE, and SSIM.

        Args:
            mu: Predicted velocity model (on GPU)
            mu_true: Ground truth velocity model (on GPU)

        Returns:
            Tuple of (mae_per_sample, rmse_per_sample, ssim_per_sample) all on GPU
        """
        batch_size = mu.shape[0]
        device = mu.device

        vm_sample_unnorm = mu.detach()
        vm_data_unnorm = v_normalize(mu_true).to(device)

        mae_per_sample = torch.mean(torch.abs(vm_sample_unnorm - vm_data_unnorm), dim=(1, 2, 3))
        mse_per_sample = torch.mean((vm_sample_unnorm - vm_data_unnorm) ** 2, dim=(1, 2, 3))
        rmse_per_sample = torch.sqrt(mse_per_sample)

        vm_sample_01 = (vm_sample_unnorm + 1) / 2
        vm_data_01 = (vm_data_unnorm + 1) / 2

        ssim_per_sample = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            ssim_val = self.ssim_loss(vm_sample_01[i:i + 1], vm_data_01[i:i + 1])
            ssim_per_sample[i] = ssim_val

        return (mae_per_sample, rmse_per_sample, ssim_per_sample)
