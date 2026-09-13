# Turbine Project

An end-to-end, federated anomaly detection pipeline for wind turbine SCADA data, built and validated against the real [CARE-to-Compare](https://www.edp.com/en/innovation/open-data/data) dataset (wind farms A, B, C).

## What is implemented

- Schema-agnostic ingestion: auto-detects the CSV delimiter and matches sensor columns by physical role/statistic (power, wind speed, reactive power) instead of hardcoded column names, so it works across exports that number sensors differently.
- Whole-farm directory ingestion: a farm's many per-event CSVs (e.g. 22 files for Farm A) are ingested as one dataset, accumulating incrementally without overwriting prior progress.
- Metadata-aware feature engineering: when a farm's `feature_description.csv` is present, anonymized `sensor_N` columns are grouped by what they actually measure (temperature, rotational speed, electrical, angle) instead of one undifferentiated average, and every group is standardized before aggregating so no single sensor's scale can dominate.
- Correct healthy/fault status semantics per the dataset's own documented status codes, with ambiguous states (e.g. Idling, Derated, Other) excluded from the training baseline and from strict accuracy metrics rather than guessed either way.
- Local per-turbine anomaly detection using a multivariate `IsolationForest`, with row-level threshold calibration and single-row prediction flicker suppressed before event-level scoring.
- Federated server-side aggregation into a manager-safe, tree-weighted FedAvg-style global ensemble model.
- Evaluation at row, turbine, and fault-event levels, with plain-text per-farm summary reports alongside the JSON metrics.
- AWS Lambda inference entrypoint with optional SNS alert publishing.
- Drift monitoring outputs, a Streamlit dashboard, Docker-based local deployment, and CI-style automation scripts.
- Tests, config-driven execution, JSON reports, and prediction exports.

## Design notes

`IsolationForest` is the core anomaly detector: unsupervised training on confirmed-healthy operation, with thresholds calibrated (not fit) against the dataset's real operational status codes. Federated aggregation is tree-weighted FedAvg-style rather than literal parameter averaging, since IsolationForest has no such thing as an "average tree" - it's described that way deliberately rather than overclaiming.

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

- [Executive Summary](docs/executive_summary.md)
- [Architecture Diagrams](docs/architecture.md)

## Evaluation notes

- Healthy/fault ground truth comes from `status_type_id` (`healthy_labels`/`fault_labels` in config), per the CARE-to-Compare README's documented status codes - not "anything non-zero." Statuses in neither list are ambiguous and excluded from strict metrics.
- If a turbine's prediction window contains no confirmed ground truth at all, the pipeline falls back to synthetic anomaly injection for precision, recall, and F1 estimation (`evaluation_mode: synthetic_injection` in the metrics).
- Event-level precision/recall/F1 are computed from contiguous anomaly windows per turbine, after suppressing predicted-anomaly runs shorter than `training.min_anomaly_run_length`.
- Communication reduction is the ratio between total raw ingested bytes and total local model artifact size.
- The AWS monthly cost estimate is pinned to a small EC2/Lambda footprint.
