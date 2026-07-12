from __future__ import annotations

import torch

EPSILON = 1e-8


def masked_squared_error_sum(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return (((predictions - targets) ** 2) * masks).sum()


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return masked_squared_error_sum(predictions, targets, masks) / (masks.sum() + EPSILON)


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(masked_mse_loss(predictions, targets, masks))
