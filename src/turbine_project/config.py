from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    csv_path: Path
    processed_dir: Path
    reports_dir: Path
    chunk_size: int
    feature_columns: list[str]
    timestamp_column: str
    asset_id_column: str
    split_column: str
    label_column: str
    split_train_value: str
    split_test_value: str
    healthy_label: int
    low_quantile: float
    high_quantile: float
    event_gap_minutes: int
    min_required_columns: list[str]
    csv_delimiter: str = "auto"
    # status_type_id values with confirmed ground truth. Anything not in
    # either list (e.g. "Derated"/"Other" per the CARE-to-Compare README) is
    # ambiguous and excluded from training's healthy baseline and from
    # strict accuracy metrics, rather than guessed either way. "Idling" is
    # deliberately NOT in the healthy default - operationally it isn't a
    # fault, but its sensor readings haven't been validated as resembling
    # normal production, so it stays excluded until that's confirmed.
    healthy_labels: list[int] = None  # type: ignore[assignment]
    fault_labels: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.healthy_labels is None:
            self.healthy_labels = [0]
        if self.fault_labels is None:
            self.fault_labels = [3, 4]


@dataclass(slots=True)
class TrainingConfig:
    contamination: float
    random_state: int
    n_estimators: int
    max_samples: str | int | float
    model_dir: Path
    synthetic_eval_multiplier: float
    synthetic_eval_seed: int
    calibration_fraction: float
    threshold_grid_size: int
    aggregation_target_estimators: int
    # Suppresses predicted-anomaly runs shorter than this many consecutive
    # rows (at 10-minute cadence, 3 rows = 30 minutes) before scoring or
    # saving predictions. Real faults last for days per the CARE-to-Compare
    # README; single-row flags are noise that fragments one true detection
    # into hundreds of spurious "events" without this.
    min_anomaly_run_length: int = 3


@dataclass(slots=True)
class FederatedConfig:
    rounds: int
    min_clients: int
    server_address: str
    global_model_path: Path


@dataclass(slots=True)
class InferenceConfig:
    predictions_dir: Path
    plots_dir: Path
    metrics_dir: Path
    lambda_model_path: Path
    sns_topic_arn: str
    dashboard_dir: Path
    monitoring_dir: Path


@dataclass(slots=True)
class AppConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    federated: FederatedConfig
    inference: InferenceConfig


def _expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_config(config_path: str | Path) -> AppConfig:
    path = _expand_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    dataset = raw["dataset"]
    training = raw["training"]
    federated = raw["federated"]
    inference = raw["inference"]

    # Environment variables override YAML for the values most likely to
    # differ between machines/environments (a laptop, CI, Streamlit Cloud),
    # without requiring a config file edit for each one.
    csv_path = os.environ.get("TURBINE_CSV_PATH", dataset["csv_path"])

    return AppConfig(
        dataset=DatasetConfig(
            csv_path=_expand_path(csv_path),
            processed_dir=_expand_path(dataset["processed_dir"]),
            reports_dir=_expand_path(dataset["reports_dir"]),
            chunk_size=int(dataset["chunk_size"]),
            feature_columns=list(dataset["feature_columns"]),
            timestamp_column=str(dataset["timestamp_column"]),
            asset_id_column=str(dataset["asset_id_column"]),
            split_column=str(dataset["split_column"]),
            label_column=str(dataset["label_column"]),
            split_train_value=str(dataset["split_train_value"]),
            split_test_value=str(dataset["split_test_value"]),
            healthy_label=int(dataset["healthy_label"]),
            healthy_labels=[int(value) for value in dataset["healthy_labels"]] if "healthy_labels" in dataset else None,
            fault_labels=[int(value) for value in dataset["fault_labels"]] if "fault_labels" in dataset else None,
            low_quantile=float(dataset["low_quantile"]),
            high_quantile=float(dataset["high_quantile"]),
            event_gap_minutes=int(dataset["event_gap_minutes"]),
            min_required_columns=list(dataset["min_required_columns"]),
            csv_delimiter=str(dataset.get("csv_delimiter", "auto")),
        ),
        training=TrainingConfig(
            contamination=float(training["contamination"]),
            random_state=int(training["random_state"]),
            n_estimators=int(training["n_estimators"]),
            max_samples=training["max_samples"],
            model_dir=_expand_path(training["model_dir"]),
            synthetic_eval_multiplier=float(training["synthetic_eval_multiplier"]),
            synthetic_eval_seed=int(training["synthetic_eval_seed"]),
            calibration_fraction=float(training["calibration_fraction"]),
            threshold_grid_size=int(training["threshold_grid_size"]),
            aggregation_target_estimators=int(training["aggregation_target_estimators"]),
            min_anomaly_run_length=int(training.get("min_anomaly_run_length", 3)),
        ),
        federated=FederatedConfig(
            rounds=int(federated["rounds"]),
            min_clients=int(federated["min_clients"]),
            server_address=str(federated["server_address"]),
            global_model_path=_expand_path(federated["global_model_path"]),
        ),
        inference=InferenceConfig(
            predictions_dir=_expand_path(inference["predictions_dir"]),
            plots_dir=_expand_path(inference["plots_dir"]),
            metrics_dir=_expand_path(inference["metrics_dir"]),
            lambda_model_path=_expand_path(inference["lambda_model_path"]),
            sns_topic_arn=str(inference["sns_topic_arn"]),
            dashboard_dir=_expand_path(inference["dashboard_dir"]),
            monitoring_dir=_expand_path(inference["monitoring_dir"]),
        ),
    )
