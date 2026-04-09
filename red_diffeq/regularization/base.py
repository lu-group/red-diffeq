import torch
from typing import Optional
from red_diffeq.regularization.diffusion import RED_DiffEq
from red_diffeq.regularization.benchmark import total_variation_loss, tikhonov_loss


class RegularizationMethod:

    def __init__(self, regularization_type: Optional[str], diffusion_model=None,
                 use_time_weight: bool = False, sigma_x0: float = 0.0001,
                 fixed_timestep: int = None):
        self.regularization_type = regularization_type
        self.diffusion_model = diffusion_model
        self.use_time_weight = use_time_weight
        self.sigma_x0 = sigma_x0
        self.fixed_timestep = fixed_timestep
        if regularization_type == 'diffusion':
            self.red_diffeq = RED_DiffEq(diffusion_model, use_time_weight=use_time_weight,
                                          sigma_x0=sigma_x0, fixed_timestep=fixed_timestep)

    def get_reg_loss(self, mu: torch.Tensor, generator: Optional[torch.Generator] = None):
        if self.regularization_type == 'diffusion':
            if self.diffusion_model is None:
                raise ValueError("Diffusion model required for 'diffusion' regularization")

            height = mu.shape[2]
            width = mu.shape[3]

            if width > self.red_diffeq.input_size or height > self.red_diffeq.input_size:
                reg_loss, _, time_tensor = self.red_diffeq.get_reg_loss_patched(mu, generator=generator)
            else:
                reg_loss, _, time_tensor = self.red_diffeq.get_reg_loss(mu, generator=generator)

            return reg_loss, time_tensor

        elif self.regularization_type == 'l2':
            reg_loss = tikhonov_loss(mu)
            # Return None for timestep when not using diffusion
            return reg_loss, None

        elif self.regularization_type == 'tv':
            reg_loss = total_variation_loss(mu)
            # Return None for timestep when not using diffusion
            return reg_loss, None

        else:
            reg_loss = torch.zeros(mu.shape[0], device=mu.device, dtype=mu.dtype)
            # Return None for timestep when not using diffusion
            return reg_loss, None
