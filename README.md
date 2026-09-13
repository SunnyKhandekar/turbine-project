# Turbine Project

This repository contains an end-to-end anomaly detection pipeline for wind turbine SCADA data based on the document `2330664-PGDM-1.pdf`.

## What is implemented

- Chunked preprocessing of the raw 12.8 GB CSV into 50,000-row Parquet partitions per turbine (`asset_id`).
- Schema validation, timestamp normalization, missing-value imputation, interpolation, and quantile clipping.
- Local per-turbine anomaly detection using a multivariate `IsolationForest` with engineered power, wind, and reactive-power features.
- Semi-supervised threshold calibration using operational status codes, while keeping the anomaly detector itself unsupervised.
- Federated server-side aggregation into a manager-safe, tree-weighted FedAvg-style global ensemble model.
- Evaluation at row, turbine, and fault-event levels using `status_type_id` discovered in the dataset (`0` healthy, non-zero fault-like states).
- AWS Lambda inference entrypoint with optional SNS alert publishing.
- Drift monitoring outputs, local dashboarding, Docker-based local deployment, and CI-style automation scripts.
- Tests, config-driven execution, JSON reports, and prediction exports.

## Why the implementation looks this way

The PDF is internally inconsistent in two places: some sections describe an unsupervised `IsolationForest` pipeline, while a smaller implementation section references a TensorFlow binary classifier. The dominant algorithmic choice throughout the document is `IsolationForest`, so this repository treats it as the production path. For federated learning, the implementation uses a tree-weighted aggregation strategy that behaves like FedAvg at the client contribution level while respecting the realities of tree-based models.

## Project structure

- `configs/default.yaml`: main runtime configuration
- `docs/executive_summary.md`: manager-facing project summary
- `docs/architecture.md`: architecture diagrams and design rationale
- `src/turbine_project/preprocessing.py`: chunked preprocessing
- `src/turbine_project/feature_engineering.py`: derived physical features for explainability
- `src/turbine_project/training.py`: local training, aggregation, evaluation
- `src/turbine_project/dashboard.py`: local stakeholder dashboard
- `src/turbine_project/aws_inference.py`: Lambda inference handler
- `scripts/`: thin wrappers for common commands
- `tests/`: unit tests
- `deployment/aws_lambda.md`: deployment notes

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run the pipeline

Full run:

```powershell
.\.venv\Scripts\python -m turbine_project.cli run-all --config configs/default.yaml
```

Stage by stage:

```powershell
.\.venv\Scripts\python -m turbine_project.cli preprocess --config configs/default.yaml
.\.venv\Scripts\python -m turbine_project.cli train --config configs/default.yaml
.\.venv\Scripts\python -m turbine_project.cli aggregate --config configs/default.yaml
.\.venv\Scripts\python -m turbine_project.cli evaluate --config configs/default.yaml
```

Fast validation run on a limited sample:

```powershell
.\.venv\Scripts\python -m turbine_project.cli run-all --config configs/default.yaml --max-input-chunks 2 --max-assets 2
```

## Running a full CARE-to-Compare wind farm

The CARE-to-Compare dataset ships one CSV per turbine/event (not one combined
file per farm), so use `--source-dir` to ingest every file in a farm's
`datasets` folder in one pass. `configs/farm_a.yaml`, `configs/farm_b.yaml`,
and `configs/farm_c.yaml` are pre-set with isolated output directories per
farm (Farm C also uses a smaller chunk size, since it has 957 raw columns).

Each farm's `feature_description.csv` (ships alongside `datasets/` in every
CARE-to-Compare farm) is auto-detected and used to group anonymized
"sensor_N" columns by what they actually measure (temperature, rotational
speed, electrical) instead of blending everything into one undifferentiated
average - this matters because those columns can span wildly different
magnitudes (e.g. a cumulative energy meter in the hundreds of thousands vs.
a temperature reading in the tens), and a plain average lets the
largest-magnitude column dominate. If no `feature_description.csv` is found,
the pipeline falls back to the generic bucket with a logged warning.

```powershell
.\.venv\Scripts\python -m turbine_project.cli run-all `
  --config configs/farm_a.yaml `
  --source-dir "F:\CARE_To_Compare\Wind Farm A\Wind Farm A\datasets" `
  --report-txt outputs/reports/farm_a_summary.txt `
  --farm-name "Wind Farm A"
```

Repeat with `configs/farm_b.yaml` / `configs/farm_c.yaml` and the matching
`datasets` folder for each farm. This writes:
- `data/processed/farm_<x>/` - cleaned per-turbine Parquet
- `models/farm_<x>/` - local + global models
- `outputs/farm_<x>/` - predictions, plots, monitoring, metrics JSON
- `outputs/reports/farm_<x>_summary.txt` - the plain-text run summary

Re-running `--source-dir` against the same `--config` accumulates additional
files into existing per-turbine Parquet chunks rather than overwriting them,
so a farm can be ingested incrementally (e.g. a handful of files at a time)
without losing earlier progress.

To ingest a single file instead of a whole directory, omit `--source-dir` and
set `dataset.csv_path` in the config (or the `TURBINE_CSV_PATH` environment
variable) as usual.

Run the local dashboard:

```powershell
.\.venv\Scripts\streamlit run src/turbine_project/dashboard_entry.py
```

Run the local CI-style workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/local_ci.ps1
```

Run the local Docker deployment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1
```

## Outputs

- `data/processed/asset_id=*/chunk_*.parquet`
- `models/local/asset_*.joblib`
- `models/global/federated_isolation_forest.joblib`
- `outputs/predictions/*.parquet`
- `outputs/metrics/*.json`
- `outputs/monitoring/*.json`
- `outputs/dashboard/`
- `outputs/reports/preprocessing_report.json`

## Executive View

- [Executive Summary](F:\codes\turbine-project\docs\executive_summary.md)
- [Architecture Diagrams](F:\codes\turbine-project\docs\architecture.md)

## Evaluation notes

- The project infers `status_type_id=0` as healthy and treats non-zero states as fault-like operational events.
- If labels are unavailable or degenerate, the pipeline falls back to synthetic anomaly injection for precision, recall, and F1 estimation.
- Event-level precision/recall/F1 are computed from contiguous anomaly windows per turbine.
- Communication reduction is approximated as the ratio between raw dataset size and total local model artifact size.
- The AWS monthly cost estimate is pinned to a small EC2/Lambda footprint consistent with the PDF's sub-$12 target.
