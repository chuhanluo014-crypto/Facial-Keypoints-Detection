from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset


KEYPOINT_COLUMNS = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]

HORIZONTAL_FLIP_INDEX_PAIRS = [
    (0, 2),
    (1, 3),
    (4, 8),
    (5, 9),
    (6, 10),
    (7, 11),
    (12, 16),
    (13, 17),
    (14, 18),
    (15, 19),
    (22, 24),
    (23, 25),
]


class FacialKeypointsDataset(Dataset):
    """Dataset for Kaggle facial keypoints CSV rows."""

    def __init__(
        self,
        frame: pd.DataFrame,
        is_train: bool = False,
        flip_probability: float = 0.5,
    ) -> None:
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be between 0.0 and 1.0")
        self.frame = frame.reset_index(drop=True)
        self.is_train = is_train
        self.flip_probability = flip_probability

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = np.fromstring(row["Image"], sep=" ", dtype=np.float32).reshape(96, 96)
        keypoints = row[KEYPOINT_COLUMNS].to_numpy(dtype=np.float32)

        if self.is_train and torch.rand(()) < self.flip_probability:
            image = np.fliplr(image)
            keypoints[0::2] = 96.0 - keypoints[0::2]
            for left_index, right_index in HORIZONTAL_FLIP_INDEX_PAIRS:
                keypoints[left_index], keypoints[right_index] = (
                    keypoints[right_index],
                    keypoints[left_index],
                )

        image = image.reshape(1, 96, 96) / 255.0
        mask = ~np.isnan(keypoints)
        keypoints = np.nan_to_num(keypoints, nan=0.0)

        return (
            torch.tensor(image.copy(), dtype=torch.float32),
            torch.tensor(keypoints, dtype=torch.float32),
            torch.tensor(mask.astype(np.float32), dtype=torch.float32),
        )


class FacialKeypointsTestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        row = self.frame.iloc[index]
        image = np.fromstring(row["Image"], sep=" ", dtype=np.float32).reshape(1, 96, 96)
        image = image / 255.0
        return int(row["ImageId"]), torch.tensor(image.copy(), dtype=torch.float32)


def load_training_frame(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def create_cv_loaders(
    csv_path: str | Path,
    n_splits: int = 5,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    max_rows: int | None = None,
) -> list[tuple[DataLoader, DataLoader]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    frame = load_training_frame(csv_path)
    if max_rows is not None:
        frame = frame.head(max_rows).copy()

    loaders: list[tuple[DataLoader, DataLoader]] = []

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, val_index in splitter.split(frame):
        train_dataset = FacialKeypointsDataset(frame.iloc[train_index], is_train=True)
        val_dataset = FacialKeypointsDataset(frame.iloc[val_index], is_train=False)
        loaders.append(
            (
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                ),
                DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                ),
            )
        )

    return loaders


def create_test_loader(
    csv_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
) -> DataLoader:
    frame = pd.read_csv(csv_path)
    return DataLoader(
        FacialKeypointsTestDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
