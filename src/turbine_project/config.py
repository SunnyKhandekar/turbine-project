from __future__ import annotations

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
    anomaly_labels: list[int]
    low_quantile: float
    high_quantile: float
    event_gap_minutes: int
    min_required_columns: list[str]


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

    return AppConfig(
        dataset=DatasetConfig(
            csv_path=_expand_path(dataset["csv_path"]),
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
            anomaly_labels=[int(value) for value in dataset["anomaly_labels"]],
            low_quantile=float(dataset["low_quantile"]),
            high_quantile=float(dataset["high_quantile"]),
            event_gap_minutes=int(dataset["event_gap_minutes"]),
            min_required_columns=list(dataset["min_required_columns"]),
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
