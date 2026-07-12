from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from facial_keypoints.config import TrainingConfig
from facial_keypoints.engine import run_cross_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train facial keypoints models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.json",
        help="Path to a JSON training config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--max-rows", type=int, default=None, help="Use a small data subset.")
    parser.add_argument("--device", type=str, default=None, help="Override device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_json(args.config).resolve_paths(PROJECT_ROOT)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_rows is not None:
        config.max_rows = args.max_rows
    if args.device is not None:
        config.device = args.device

    metrics = run_cross_validation(
        train_csv=config.train_csv,
        output_dir=config.output_dir,
        model_name=config.model_name,
        pretrained=config.pretrained,
        n_splits=config.n_splits,
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        num_workers=config.num_workers,
        max_rows=config.max_rows,
        device_name=config.device,
    )

    metrics_path = config.output_dir / f"{config.model_name}_cv_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
