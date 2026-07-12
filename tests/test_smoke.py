from __future__ import annotations

import torch

from facial_keypoints.metrics import compute_rmse, masked_mse_loss
from facial_keypoints.models import SimpleCNNBaseline


def test_baseline_forward_shape() -> None:
    model = SimpleCNNBaseline()
    outputs = model(torch.zeros(2, 1, 96, 96))
    assert outputs.shape == (2, 30)


def test_masked_metrics_ignore_missing_values() -> None:
    predictions = torch.tensor([[1.0, 10.0]])
    targets = torch.tensor([[3.0, 0.0]])
    masks = torch.tensor([[1.0, 0.0]])

    loss = masked_mse_loss(predictions, targets, masks)
    rmse = compute_rmse(predictions, targets, masks)

    assert torch.isclose(loss, torch.tensor(4.0))
    assert torch.isclose(rmse, torch.tensor(2.0))
