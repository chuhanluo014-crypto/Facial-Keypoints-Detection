from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .data import create_test_loader
from .models import build_model


@torch.no_grad()
def predict_keypoints(
    checkpoint_path: str | Path,
    test_csv: str | Path,
    batch_size: int = 128,
    device_name: str = "auto",
) -> pd.DataFrame:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
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
    feature_names = [
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
    feature_to_column = {name: f"kp_{index}" for index, name in enumerate(feature_names)}
    merged = lookup.merge(predictions, on="ImageId", how="left")
    merged["Location"] = merged.apply(
        lambda row: row[feature_to_column[row["FeatureName"]]],
        axis=1,
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["RowId", "Location"]].to_csv(output_path, index=False)
    return output_path
