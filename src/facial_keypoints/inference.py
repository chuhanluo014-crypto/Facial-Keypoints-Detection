from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .data import KEYPOINT_COLUMNS, create_test_loader
from .models import build_model


@torch.no_grad()
def predict_keypoints(
    checkpoint_path: str | Path,
    test_csv: str | Path,
    batch_size: int = 128,
    device_name: str = "auto",
) -> pd.DataFrame:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu"
    )
    if device_name != "auto":
        device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        model_name=checkpoint["model_name"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows: list[dict[str, float | int]] = []
    loader = create_test_loader(test_csv, batch_size=batch_size)
    for image_ids, images in loader:
        outputs = model(images.to(device)).cpu().numpy()
        for image_id, prediction in zip(image_ids.tolist(), outputs, strict=True):
            rows.append(
                {
                    "ImageId": image_id,
                    **{f"kp_{index}": float(value) for index, value in enumerate(prediction)},
                }
            )

    return pd.DataFrame(rows)


def build_kaggle_submission(
    predictions: pd.DataFrame,
    lookup_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    lookup = pd.read_csv(lookup_csv)
    feature_to_column = {name: f"kp_{index}" for index, name in enumerate(KEYPOINT_COLUMNS)}
    unknown_features = sorted(set(lookup["FeatureName"]) - set(feature_to_column))
    if unknown_features:
        raise ValueError(f"Unsupported lookup features: {unknown_features}")

    prediction_index = predictions.set_index("ImageId")
    locations = [
        prediction_index.at[image_id, feature_to_column[feature_name]]
        for image_id, feature_name in zip(
            lookup["ImageId"],
            lookup["FeatureName"],
            strict=True,
        )
    ]
    submission = lookup.assign(Location=locations)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission[["RowId", "Location"]].to_csv(output_path, index=False)
    return output_path
