from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from facial_keypoints.inference import build_kaggle_submission, predict_keypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Kaggle submission CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "test.csv",
    )
    parser.add_argument(
        "--lookup-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "IdLookupTable.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "submission.csv",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_keypoints(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    output_path = build_kaggle_submission(
        predictions=predictions,
        lookup_csv=args.lookup_csv,
        output_csv=args.output_csv,
    )
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
