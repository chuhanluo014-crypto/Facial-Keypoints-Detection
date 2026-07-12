from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from .data import create_cv_loaders
from .metrics import EPSILON, masked_mse_loss, masked_squared_error_sum
from .models import build_model


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_squared_error = 0.0
    total_observed = 0.0

    for images, keypoints, masks in loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = masked_mse_loss(outputs, keypoints, masks)
        loss.backward()
        optimizer.step()

        total_squared_error += masked_squared_error_sum(
            outputs.detach(),
            keypoints,
            masks,
        ).item()
        total_observed += masks.sum().item()

    return total_squared_error / max(total_observed, EPSILON)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_squared_error = 0.0
    total_observed = 0.0

    for images, keypoints, masks in loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        masks = masks.to(device)

        outputs = model(images)
        total_squared_error += masked_squared_error_sum(outputs, keypoints, masks).item()
        total_observed += masks.sum().item()

    return float(np.sqrt(total_squared_error / max(total_observed, EPSILON)))


def run_cross_validation(
    train_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "baseline_cnn",
    pretrained: bool = False,
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    seed: int = 42,
    num_workers: int = 0,
    max_rows: int | None = None,
    device_name: str = "auto",
) -> dict[str, float | list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loaders = create_cv_loaders(
        csv_path=train_csv,
        n_splits=n_splits,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        max_rows=max_rows,
    )
    device = resolve_device(device_name)
    fold_rmses: list[float] = []

    for fold_index, (train_loader, val_loader) in enumerate(loaders, start=1):
        model = build_model(model_name=model_name, pretrained=pretrained).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_rmse = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_rmse = evaluate(model, val_loader, device)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state_dict = copy.deepcopy(model.state_dict())
            print(
                f"fold={fold_index}/{n_splits} "
                f"epoch={epoch}/{epochs} "
                f"train_mse={train_loss:.4f} "
                f"val_rmse={val_rmse:.4f}"
            )

        fold_rmses.append(best_rmse)
        checkpoint = {
            "model_name": model_name,
            "pretrained": pretrained,
            "state_dict": best_state_dict,
            "best_rmse": best_rmse,
            "fold": fold_index,
        }
        torch.save(checkpoint, output_path / f"{model_name}_fold{fold_index}.pt")

    return {
        "fold_rmses": fold_rmses,
        "mean_rmse": float(np.mean(fold_rmses)),
        "std_rmse": float(np.std(fold_rmses)),
    }
