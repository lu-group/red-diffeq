import torch
import torch.nn as nn
from typing import Optional

from red_diffeq.regularization.base import RegularizationMethod

class LossCalculator:
    """Calculate observation and regularization losses for FWI optimization."""

    def __init__(self, regularization_method: RegularizationMethod):
        self.regularization_method = regularization_method

    def observation_loss(self, predicted: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute observation loss (data fidelity term).

        Args:
            predicted: Predicted seismic data
            target: Observed seismic data
            mask: Optional mask tensor (1 for observed data, 0 for missing).
                  If provided, loss is only computed on observed (non-missing) data.

        Returns:
            Per-model observation loss (batch_size,)
        """
        loss = nn.L1Loss(reduction='none')(target.float(), predicted.float())

        if mask is not None:
            loss = loss * mask
            num_observed = mask.sum(dim=tuple(range(1, len(mask.shape)))).clamp(min=1.0)
            loss = loss.sum(dim=tuple(range(1, len(loss.shape)))) / num_observed
        else:
            loss = loss.mean(dim=tuple(range(1, len(loss.shape))))

        return loss

    def regularization_loss(self, mu: torch.Tensor, generator: Optional[torch.Generator] = None):
        """Compute regularization loss.

        Args:
            mu: Velocity model (batch, 1, height, width)
            generator: Optional torch.Generator for deterministic noise sampling

        Returns:
            Tuple of (per-model regularization loss (batch_size,), diffusion timestep tensor or None)
        """
        return self.regularization_method.get_reg_loss(mu, generator=generator)

    def total_loss(self, obs_loss: torch.Tensor, reg_loss: torch.Tensor, reg_lambda: float) -> torch.Tensor:
        """Compute total loss = observation + λ * regularization.

        Args:
            obs_loss: Observation loss (batch_size,)
            reg_loss: Regularization loss (batch_size,)
            reg_lambda: Regularization weight

        Returns:
            Total loss (batch_size,)
        """
        return obs_loss + reg_lambda * reg_loss
