import torch


def total_variation_loss(mu: torch.Tensor) -> torch.Tensor:
    """Compute Total Variation (TV) regularization.

    Args:
        mu: Velocity model (batch, 1, height, width)

    Returns:
        Per-model TV loss (batch_size,)
    """
    diff_x = torch.abs(mu[:, :, :, 1:] - mu[:, :, :, :-1])
    diff_y = torch.abs(mu[:, :, 1:, :] - mu[:, :, :-1, :])

    tv_x = diff_x.view(diff_x.shape[0], -1).mean(dim=1)
    tv_y = diff_y.view(diff_y.shape[0], -1).mean(dim=1)

    return tv_x + tv_y


def tikhonov_loss(mu: torch.Tensor) -> torch.Tensor:
    """Compute Tikhonov (L2 smoothness) regularization.

    Args:
        mu: Velocity model (batch, 1, height, width)

    Returns:
        Per-model L2 loss (batch_size,)
    """
    diff_x = mu[:, :, :, 1:] - mu[:, :, :, :-1]
    diff_y = mu[:, :, 1:, :] - mu[:, :, :-1, :]

    l2_x = (diff_x ** 2).view(diff_x.shape[0], -1).mean(dim=1)
    l2_y = (diff_y ** 2).view(diff_y.shape[0], -1).mean(dim=1)

    return l2_x + l2_y
