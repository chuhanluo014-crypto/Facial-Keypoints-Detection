from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    train_csv: Path = Path("data/raw/training.csv")
    test_csv: Path = Path("data/raw/test.csv")
    lookup_csv: Path = Path("data/raw/IdLookupTable.csv")
    output_dir: Path = Path("outputs")
    model_name: str = "baseline_cnn"
    pretrained: bool = False
    n_splits: int = 5
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    max_rows: int | None = None
    device: str = "auto"

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw: dict[str, Any] = json.load(file)

        path_fields = {"train_csv", "test_csv", "lookup_csv", "output_dir"}
        normalized = {
            key: Path(value) if key in path_fields else value
            for key, value in raw.items()
        }
        return cls(**normalized)

    def resolve_paths(self, project_root: str | Path) -> "TrainingConfig":
        root = Path(project_root)
        resolved = self.__dict__.copy()
        for key in ("train_csv", "test_csv", "lookup_csv", "output_dir"):
            value = Path(resolved[key])
            resolved[key] = value if value.is_absolute() else root / value
        return TrainingConfig(**resolved)
