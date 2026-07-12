# SDSC8007 Facial Keypoints Detection

Production-style computer vision project for facial landmark regression. The project predicts 15 facial keypoints from 96x96 grayscale face images using PyTorch models, including a simple CNN baseline and a MobileNetV2-based transfer-learning model.

## Why This Structure

The repository has been organized like an algorithm engineering deliverable rather than a one-off course folder:

- `src/facial_keypoints/`: reusable data, model, metric, training, and inference code.
- `scripts/`: command-line entry points for training and submission generation.
- `configs/`: experiment configs for baseline CNN and MobileNetV2.
- `data/raw/`: immutable raw CSV data tracked with Git LFS.
- `notebooks/`: exploratory notebook preserved as experiment history.
- `docs/`: report, presentation, assignment brief, and templates.
- `tests/`: smoke tests for model shape and masked metric behavior.
- `outputs/`: generated checkpoints, metrics, and submissions.

## Repository Layout

```text
.
|-- configs/
|   |-- baseline.json
|   `-- mobilenet_v2.json
|-- data/
|   |-- README.md
|   `-- raw/
|       |-- IdLookupTable.csv
|       |-- SampleSubmission.csv
|       |-- test.csv
|       `-- training.csv
|-- docs/
|   |-- project_brief/
|   |-- presentations/
|   `-- reports/
|-- notebooks/
|   `-- facial_keypoints_experiment.ipynb
|-- scripts/
|   |-- make_submission.py
|   `-- train.py
|-- src/
|   `-- facial_keypoints/
|-- tests/
`-- outputs/
```

## Environment

Create an isolated environment before running experiments.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA training, install the CUDA-enabled PyTorch build recommended by the official PyTorch selector first, then install the remaining dependencies.

## Data And Git LFS

Large CSV and binary document files are tracked through Git LFS.

```powershell
git lfs install
git lfs pull
```

Raw data should stay in `data/raw/`. Put generated or cleaned files in `data/processed/`, which is intentionally ignored by Git.

## Training

Run the baseline CNN:

```powershell
python scripts/train.py --config configs/baseline.json
```

Run the MobileNetV2 experiment:

```powershell
python scripts/train.py --config configs/mobilenet_v2.json
```

Short smoke run:

```powershell
python scripts/train.py --config configs/baseline.json --epochs 1 --batch-size 64 --max-rows 256
```

Training writes fold checkpoints and cross-validation metrics under `outputs/`.

## Submission

After training, generate a Kaggle-style submission from a checkpoint:

```powershell
python scripts/make_submission.py --checkpoint outputs/baseline_cnn/baseline_cnn_fold1.pt
```

The submission is saved to `outputs/submission.csv` by default.

## Testing

```powershell
pytest
```

The tests are intentionally lightweight and verify the baseline model output shape plus masked metric behavior.

## Model Notes

- Baseline: 3-block CNN with 30-dimensional coordinate regression output.
- Improved model: MobileNetV2 adapted for one-channel grayscale input and a 30-coordinate regression head.
- Objective: masked MSE to handle missing labels.
- Metric: RMSE, aligned with the facial keypoints benchmark convention.
