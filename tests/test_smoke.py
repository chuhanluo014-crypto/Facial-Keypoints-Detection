from __future__ import annotations

import pandas as pd
import torch

from facial_keypoints.data import KEYPOINT_COLUMNS, FacialKeypointsDataset
from facial_keypoints.metrics import compute_rmse, masked_mse_loss, masked_squared_error_sum
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


def test_masked_squared_error_sum_counts_observed_values() -> None:
    predictions = torch.tensor([[1.0, 10.0], [5.0, 5.0]])
    targets = torch.tensor([[3.0, 0.0], [1.0, 1.0]])
    masks = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    assert torch.isclose(
        masked_squared_error_sum(predictions, targets, masks),
        torch.tensor(36.0),
    )


def test_horizontal_flip_swaps_left_and_right_landmarks() -> None:
    row = {column: float(index) for index, column in enumerate(KEYPOINT_COLUMNS)}
    row["Image"] = " ".join(["0"] * (96 * 96))
    dataset = FacialKeypointsDataset(
        pd.DataFrame([row]),
        is_train=True,
        flip_probability=1.0,
    )

    _, keypoints, mask = dataset[0]

    assert torch.all(mask == 1.0)
    assert torch.isclose(keypoints[4], torch.tensor(96.0 - 8.0))
    assert torch.isclose(keypoints[6], torch.tensor(96.0 - 10.0))
    assert torch.isclose(keypoints[12], torch.tensor(96.0 - 16.0))
    assert torch.isclose(keypoints[14], torch.tensor(96.0 - 18.0))
