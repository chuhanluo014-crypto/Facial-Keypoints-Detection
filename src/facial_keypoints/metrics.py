from __future__ import annotations

import torch


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    masked_diff = (predictions - targets) * masks
    return (masked_diff**2).sum() / (masks.sum() + 1e-8)


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return torch.sqrt(masked_mse_loss(predictions, targets, masks))
