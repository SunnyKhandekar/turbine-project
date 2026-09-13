from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from .config import AppConfig
from .models.artifacts import FederatedIsolationForestEnsemble, LocalModelArtifact, load_joblib
from .utils import ensure_directory, portable_path, read_json, write_json

LOGGER = logging.getLogger(__name__)


def _asset_chunk_paths(processed_dir: Path) -> dict[str, list[Path]]:
    assets: dict[str, list[Path]] = {}
    for asset_dir in sorted(processed_dir.glob("asset_id=*")):
        asset_id = asset_dir.name.split("=", maxsplit=1)[1]
        assets[asset_id] = sorted(asset_dir.glob("chunk_*.parquet"))
    return assets


def _load_asset_frame(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], utc=True, errors="coerce")
    return frame.sort_values("time_stamp").reset_index(drop=True)


def _label_array(frame: pd.DataFrame, dataset_config) -> np.ndarray | None:
    """Return per-row ground truth: 0 healthy, 1 fault, -1 ambiguous/unknown.

    Only status_type_id values in dataset_config.healthy_labels/fault_labels
    have confirmed ground truth; everything else (e.g. "Idling", "Other")
    is -1 and is excluded from the training healthy baseline and from
    strict accuracy metrics by callers, rather than guessed either way.
    """
    if "status_type_id" not in frame.columns:
        return None
    fallback = dataset_config.healthy_labels[0] if dataset_config.healthy_labels else dataset_config.healthy_label
    raw = pd.to_numeric(frame["status_type_id"], errors="coerce").fillna(fallback).astype(int)
    labels = np.full(len(raw), -1, dtype=int)
    labels[raw.isin(dataset_config.healthy_labels).to_numpy()] = 0
    labels[raw.isin(dataset_config.fault_labels).to_numpy()] = 1
    return labels


def _valid_mask(labels: np.ndarray | None) -> np.ndarray | None:
    return None if labels is None else labels != -1


def _suppress_short_runs(predictions: np.ndarray, min_run: int) -> np.ndarray:
    """Zero out predicted-anomaly runs shorter than min_run consecutive rows.

    Real faults last for days (per the CARE-to-Compare README), so an
    isolated single-row anomaly flag is almost certainly noise, not a real
    detection - left alone, it fragments one true fault window into hundreds
    of spurious "events" at evaluation time without changing the underlying
    anomaly score (roc_auc/average_precision, which use the continuous score
    directly, are unaffected).
    """
    if min_run <= 1 or len(predictions) == 0:
        return predictions
    predictions = predictions.astype(int)
    is_change = np.empty(len(predictions), dtype=bool)
    is_change[0] = True
    is_change[1:] = predictions[1:] != predictions[:-1]
    run_id = np.cumsum(is_change)
    run_lengths = np.bincount(run_id)[run_id]
    result = predictions.copy()
    result[(predictions == 1) & (run_lengths < min_run)] = 0
    return result


def _score_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, -scores))
        metrics["average_precision"] = float(average_precision_score(labels, -scores))
    cm = confusion_matrix(labels, predictions)
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])
    return metrics


def _extract_events(
    frame: pd.DataFrame,
    flag_column: str,
    event_gap_minutes: int,
    start_name: str,
    end_name: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[start_name, end_name])
    working = frame[["time_stamp", flag_column]].copy()
    time_gap = working["time_stamp"].diff().dt.total_seconds().div(60).fillna(0)
    group_id = ((working[flag_column].ne(working[flag_column].shift())) | (time_gap > event_gap_minutes)).cumsum()
    events = (
        working.loc[working[flag_column] == 1]
        .groupby(group_id)
        .agg(**{start_name: ("time_stamp", "min"), end_name: ("time_stamp", "max")})
        .reset_index(drop=True)
    )
    return events


def _event_frame(frame: pd.DataFrame, label_column: str, prediction_column: str, fault_labels: list[int], event_gap_minutes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame[["time_stamp", label_column, prediction_column]].copy()
    # A confirmed fault event window is defined only by status codes with
    # confirmed ground truth (fault_labels), not "anything not healthy" -
    # ambiguous statuses (e.g. Idling, Other) are simply not part of a fault
    # window rather than silently counted as one.
    working["actual_flag"] = pd.to_numeric(working[label_column], errors="coerce").isin(fault_labels).astype(int)
    working["pred_flag"] = working[prediction_column].astype(int)
    actual_events = _extract_events(working, "actual_flag", event_gap_minutes, "actual_start", "actual_end")
    pred_events = _extract_events(working, "pred_flag", event_gap_minutes, "pred_start", "pred_end")
    return actual_events, pred_events


def _event_metrics(frame: pd.DataFrame, config: AppConfig) -> dict[str, float]:
    actual_events, pred_events = _event_frame(
        frame=frame,
        label_column=config.dataset.label_column,
        prediction_column="anomaly_prediction",
        fault_labels=config.dataset.fault_labels,
        event_gap_minutes=config.dataset.event_gap_minutes,
    )
    if actual_events.empty and pred_events.empty:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0, "actual_events": 0, "predicted_events": 0}

    matched_actual = set()
    matched_pred = set()
    pred_pointer = 0
    pred_records = list(pred_events.itertuples(index=True))
    for actual in actual_events.itertuples(index=True):
        while pred_pointer < len(pred_records) and pred_records[pred_pointer].pred_end < actual.actual_start:
            pred_pointer += 1
        probe = pred_pointer
        while probe < len(pred_records) and pred_records[probe].pred_start <= actual.actual_end:
            matched_actual.add(actual.Index)
            matched_pred.add(pred_records[probe].Index)
            probe += 1

    actual_count = len(actual_events)
    predicted_count = len(pred_events)
    event_precision = len(matched_pred) / predicted_count if predicted_count else 0.0
    event_recall = len(matched_actual) / actual_count if actual_count else 0.0
    event_f1 = (2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) else 0.0
    return {
        "event_precision": float(event_precision),
        "event_recall": float(event_recall),
        "event_f1": float(event_f1),
        "actual_events": int(actual_count),
        "predicted_events": int(predicted_count),
    }


def _calibration_split(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = max(int(len(frame) * (1 - fraction)), 1)
    train_frame = frame.iloc[:cutoff].copy()
    calibration_frame = frame.iloc[cutoff:].copy()
    if calibration_frame.empty:
        calibration_frame = train_frame.copy()
    return train_frame, calibration_frame


def _fit_local_model(train_frame: pd.DataFrame, config: AppConfig) -> tuple[LocalModelArtifact, pd.DataFrame]:
    feature_names = config.dataset.feature_columns
    labels = _label_array(train_frame, config.dataset)
    # Fit only on rows with *confirmed* healthy status - ambiguous statuses
    # (e.g. Idling) are excluded rather than assumed to belong in the
    # "normal" baseline the model learns from.
    healthy_train = train_frame.loc[labels == 0].copy() if labels is not None and (labels == 1).any() else train_frame.copy()
    train_fit_frame, calibration_frame = _calibration_split(healthy_train, config.training.calibration_fraction)
    if train_fit_frame.empty:
        train_fit_frame = healthy_train.copy()
    if calibration_frame.empty:
        calibration_frame = healthy_train.copy()

    x_train = train_fit_frame[feature_names].to_numpy(dtype=float)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = IsolationForest(
        contamination=config.training.contamination,
        random_state=config.training.random_state,
        n_estimators=config.training.n_estimators,
        max_samples=config.training.max_samples,
    )
    model.fit(x_train_scaled)
    train_scores = model.decision_function(x_train_scaled)
    default_threshold = float(np.quantile(train_scores, config.training.contamination))

    threshold = default_threshold
    calibration_pool = train_frame.iloc[-max(len(calibration_frame), 1) :].copy()
    calibration_labels = _label_array(calibration_pool, config.dataset)
    valid_mask = _valid_mask(calibration_labels)
    if valid_mask is not None and valid_mask.any() and len(np.unique(calibration_labels[valid_mask])) > 1:
        x_cal = calibration_pool[feature_names].to_numpy(dtype=float)
        calibration_scores = model.decision_function(scaler.transform(x_cal))
        quantiles = np.linspace(0.01, 0.25, config.training.threshold_grid_size)
        candidate_thresholds = np.quantile(calibration_scores, quantiles)
        best_score = -1.0
        for candidate in candidate_thresholds:
            predictions = (calibration_scores < candidate).astype(int)
            predictions = _suppress_short_runs(predictions, config.training.min_anomaly_run_length)
            # Row-level F1 is graded only on confirmed labels; event
            # extraction below still runs over the full (unmasked) window
            # since it defines "fault" by status code directly, and ambiguous
            # rows simply aren't part of a fault window.
            row_f1 = f1_score(calibration_labels[valid_mask], predictions[valid_mask], zero_division=0)
            calibration_pool["anomaly_prediction"] = predictions
            event_score = _event_metrics(calibration_pool, config)["event_f1"]
            combined = 0.6 * event_score + 0.4 * row_f1
            if combined > best_score:
                best_score = combined
                threshold = float(candidate)

    artifact = LocalModelArtifact(
        asset_id=str(train_frame[config.dataset.asset_id_column].iloc[0]),
        feature_names=feature_names,
        contamination=config.training.contamination,
        sample_count=len(train_fit_frame),
        scaler=scaler,
        model=model,
        score_threshold=threshold,
        train_score_mean=float(train_scores.mean()),
        train_score_std=float(train_scores.std(ddof=0)),
        metadata={"fit_rows": int(len(train_fit_frame)), "calibration_rows": int(len(calibration_pool))},
    )
    return artifact, calibration_pool


def train_local_models(config: AppConfig, max_assets: int | None = None) -> dict[str, object]:
    asset_paths = _asset_chunk_paths(config.dataset.processed_dir)
    if not asset_paths:
        raise FileNotFoundError("No preprocessed asset chunks were found. Run preprocess first.")

    local_dir = ensure_directory(config.training.model_dir / "local")
    metrics_dir = ensure_directory(config.inference.metrics_dir)
    summary: dict[str, object] = {"assets": {}, "feature_names": config.dataset.feature_columns}

    for index, (asset_id, paths) in enumerate(asset_paths.items()):
        if max_assets is not None and index >= max_assets:
            break
        frame = _load_asset_frame(paths)
        train_mask = frame[config.dataset.split_column].astype(str).str.lower().eq(config.dataset.split_train_value.lower())
        train_frame = frame.loc[train_mask].copy()
        if train_frame.empty:
            train_frame = frame.copy()
        artifact, calibration_pool = _fit_local_model(train_frame, config)
        model_path = local_dir / f"asset_{asset_id}.joblib"
        artifact.model_path = model_path
        joblib.dump(artifact, model_path)

        calibration_scores = artifact.score(calibration_pool[config.dataset.feature_columns].to_numpy(dtype=float))
        calibration_labels = _label_array(calibration_pool, config.dataset)
        valid_mask = _valid_mask(calibration_labels)
        if valid_mask is not None and valid_mask.any() and len(np.unique(calibration_labels[valid_mask])) > 1:
            calibration_predictions = (calibration_scores < artifact.score_threshold).astype(int)
            calibration_predictions = _suppress_short_runs(calibration_predictions, config.training.min_anomaly_run_length)
            calibration_metrics = _score_metrics(
                calibration_labels[valid_mask], calibration_predictions[valid_mask], calibration_scores[valid_mask]
            )
            calibration_metrics["ambiguous_rows_excluded"] = int((~valid_mask).sum())
        else:
            calibration_metrics = {}

        summary["assets"][asset_id] = {
            "model_path": str(model_path),
            "fit_rows": artifact.metadata["fit_rows"],
            "calibration_rows": artifact.metadata["calibration_rows"],
            "threshold": artifact.score_threshold,
            "train_score_mean": artifact.train_score_mean,
            "train_score_std": artifact.train_score_std,
            "calibration_metrics": calibration_metrics,
        }
        LOGGER.info("Trained local model for asset %s with %s fit rows", asset_id, artifact.sample_count)

    write_json(metrics_dir / "local_training_summary.json", summary)
    return summary


def aggregate_local_models(config: AppConfig) -> FederatedIsolationForestEnsemble:
    local_models = sorted((config.training.model_dir / "local").glob("asset_*.joblib"))
    if not local_models:
        raise FileNotFoundError("No local models found for federated aggregation.")

    artifacts = [load_joblib(path) for path in local_models]
    tree_weights = np.array([artifact.sample_count * len(artifact.model.estimators_) for artifact in artifacts], dtype=float)
    tree_weights = tree_weights / tree_weights.sum()
    global_threshold = float(np.average([artifact.score_threshold for artifact in artifacts], weights=tree_weights))
    ensemble = FederatedIsolationForestEnsemble(
        feature_names=config.dataset.feature_columns,
        client_models=artifacts,
        weights=tree_weights,
        global_threshold=global_threshold,
    )
    ensemble.save(config.federated.global_model_path)
    write_json(
        config.inference.metrics_dir / "federated_aggregation_summary.json",
        {
            "global_model_path": str(config.federated.global_model_path),
            "client_count": len(artifacts),
            "tree_weighted_contributions": {artifact.asset_id: float(weight) for artifact, weight in zip(artifacts, tree_weights)},
            "global_threshold": global_threshold,
            "rounds": config.federated.rounds,
            "aggregation_strategy": "manager_safe_tree_weighted_fedavg_style",
        },
    )
    return ensemble


def _synthetic_labels(x: np.ndarray, multiplier: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    synthetic = x.copy()
    synthetic[:, 0] = synthetic[:, 0] + multiplier * np.maximum(np.std(x[:, 0]), 1e-6) * rng.normal(size=len(x))
    combined = np.vstack([x, synthetic])
    labels = np.concatenate([np.zeros(len(x), dtype=int), np.ones(len(synthetic), dtype=int)])
    return combined, labels


def _write_plots(asset_id: str, frame: pd.DataFrame, labels: np.ndarray | None, plots_dir: Path) -> dict[str, str]:
    ensure_directory(plots_dir)
    score_plot = plots_dir / f"asset_{asset_id}_score_distribution.png"
    plt.figure(figsize=(10, 5))
    sns.histplot(frame["anomaly_score"], bins=50, kde=True, color="steelblue")
    plt.title(f"Asset {asset_id} anomaly score distribution")
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(score_plot)
    plt.close()

    feature_plot = plots_dir / f"asset_{asset_id}_feature_scatter.png"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=frame.sample(min(len(frame), 5000), random_state=42),
        x="wind_speed_avg",
        y="power_avg",
        hue="anomaly_prediction",
        palette={0: "steelblue", 1: "darkorange"},
        alpha=0.6,
        s=20,
    )
    plt.title(f"Asset {asset_id} power vs wind speed")
    plt.tight_layout()
    plt.savefig(feature_plot)
    plt.close()

    outputs = {"score_distribution": str(score_plot), "feature_scatter": str(feature_plot)}
    valid_mask = None if labels is None else labels != -1
    if valid_mask is not None and valid_mask.any() and len(np.unique(labels[valid_mask])) > 1:
        cm = confusion_matrix(labels[valid_mask], frame.loc[valid_mask, "anomaly_prediction"])
        cm_plot = plots_dir / f"asset_{asset_id}_confusion_matrix.png"
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
        plt.title(f"Asset {asset_id} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(cm_plot)
        plt.close()
        outputs["confusion_matrix"] = str(cm_plot)
    return outputs


def _raw_dataset_bytes(config: AppConfig) -> int:
    """Total raw input bytes actually ingested, for the communication-reduction metric.

    Directory-mode ingestion (a whole farm's many per-event files) has no
    single "the raw dataset" file - config.dataset.csv_path is just an
    unused placeholder in that mode - so this reads the real total recorded
    by preprocessing into preprocessing_report.json, falling back to
    csv_path's size for the single-file workflow where that report field
    isn't present.
    """
    report_path = config.dataset.reports_dir / "preprocessing_report.json"
    if report_path.exists():
        report = read_json(report_path)
        if "total_input_bytes" in report:
            return int(report["total_input_bytes"])
    return config.dataset.csv_path.stat().st_size if config.dataset.csv_path.exists() else 0


def _monitoring_summary(train_frame: pd.DataFrame, inference_frame: pd.DataFrame, feature_names: list[str]) -> dict[str, object]:
    drift_rows = []
    for feature in feature_names:
        train_series = train_frame[feature].astype(float)
        inference_series = inference_frame[feature].astype(float)
        train_mean = float(train_series.mean())
        inference_mean = float(inference_series.mean())
        train_std = float(train_series.std(ddof=0)) or 1e-6
        shift_sigma = abs(inference_mean - train_mean) / train_std
        drift_rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "inference_mean": inference_mean,
                "sigma_shift": float(shift_sigma),
            }
        )
    return {"feature_drift": drift_rows}


def evaluate_global_model(config: AppConfig, max_assets: int | None = None) -> dict[str, object]:
    ensemble: FederatedIsolationForestEnsemble = load_joblib(config.federated.global_model_path)
    asset_paths = _asset_chunk_paths(config.dataset.processed_dir)
    predictions_dir = ensure_directory(config.inference.predictions_dir)
    metrics_dir = ensure_directory(config.inference.metrics_dir)
    plots_dir = ensure_directory(config.inference.plots_dir)
    monitoring_dir = ensure_directory(config.inference.monitoring_dir)
    all_metrics: dict[str, object] = {"assets": {}, "feature_names": config.dataset.feature_columns}
    project_root = metrics_dir.parent.parent

    for index, (asset_id, paths) in enumerate(asset_paths.items()):
        if max_assets is not None and index >= max_assets:
            break
        frame = _load_asset_frame(paths)
        train_frame = frame.loc[frame[config.dataset.split_column].astype(str).str.lower().eq(config.dataset.split_train_value.lower())].copy()
        inference_frame = frame.loc[frame[config.dataset.split_column].astype(str).str.lower().eq(config.dataset.split_test_value.lower())].copy()
        if inference_frame.empty:
            inference_frame = frame.copy()
        x_test = inference_frame[config.dataset.feature_columns].to_numpy(dtype=float)

        start = perf_counter()
        scores = ensemble.score(x_test)
        predictions = ensemble.predict(x_test)
        predictions = _suppress_short_runs(predictions, config.training.min_anomaly_run_length)
        latency_ms = ((perf_counter() - start) / max(len(inference_frame), 1)) * 1000
        inference_frame["anomaly_score"] = scores
        inference_frame["anomaly_prediction"] = predictions
        output_path = predictions_dir / f"asset_{asset_id}_predictions.parquet"
        inference_frame.to_parquet(output_path, index=False)

        labels = _label_array(inference_frame, config.dataset)
        valid_mask = _valid_mask(labels)
        if valid_mask is not None and valid_mask.any() and len(np.unique(labels[valid_mask])) > 1:
            metrics = _score_metrics(labels[valid_mask], predictions[valid_mask], scores[valid_mask])
            metrics["evaluation_mode"] = "observed_labels"
            metrics["ambiguous_rows_excluded"] = int((~valid_mask).sum())
            # Turbine-level correctness still looks at every row: "did a
            # confirmed fault occur anywhere" and "did the model flag
            # anything anywhere" are both full-data questions, independent
            # of which individual rows have confirmed ground truth.
            turbine_actual = int((labels == 1).any())
            turbine_pred = int(predictions.sum() > 0)
            metrics["turbine_level_correct"] = int(turbine_actual == turbine_pred)
        else:
            synthetic_x, synthetic_labels = _synthetic_labels(x_test, config.training.synthetic_eval_multiplier, config.training.synthetic_eval_seed)
            synthetic_scores = ensemble.score(synthetic_x)
            synthetic_predictions = ensemble.predict(synthetic_x)
            metrics = _score_metrics(synthetic_labels, synthetic_predictions, synthetic_scores)
            metrics["evaluation_mode"] = "synthetic_injection"
            metrics["turbine_level_correct"] = None

        event_metrics = _event_metrics(inference_frame, config) if labels is not None else {
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
            "actual_events": 0,
            "predicted_events": 0,
        }
        plot_paths = _write_plots(asset_id, inference_frame, labels, plots_dir)
        monitoring = _monitoring_summary(train_frame if not train_frame.empty else inference_frame, inference_frame, config.dataset.feature_columns)
        write_json(monitoring_dir / f"asset_{asset_id}_monitoring.json", monitoring)

        metrics.update(event_metrics)
        metrics["rows"] = int(len(inference_frame))
        metrics["latency_ms_per_row"] = float(latency_ms)
        metrics["prediction_path"] = portable_path(output_path, project_root)
        metrics["plots"] = {name: portable_path(Path(path), project_root) for name, path in plot_paths.items()}
        metrics["monitoring_path"] = portable_path(monitoring_dir / f"asset_{asset_id}_monitoring.json", project_root)
        all_metrics["assets"][asset_id] = metrics

    raw_data_bytes = _raw_dataset_bytes(config)
    local_model_paths = list((config.training.model_dir / "local").glob("asset_*.joblib"))
    communication_bytes = sum(path.stat().st_size for path in local_model_paths if path.exists())
    turbine_correct = [asset_metrics["turbine_level_correct"] for asset_metrics in all_metrics["assets"].values() if asset_metrics["turbine_level_correct"] is not None]
    all_metrics["portfolio_summary"] = {
        "assets_evaluated": len(all_metrics["assets"]),
        "turbine_accuracy": float(sum(turbine_correct) / len(turbine_correct)) if turbine_correct else None,
        "mean_event_f1": float(np.mean([asset_metrics["event_f1"] for asset_metrics in all_metrics["assets"].values()])) if all_metrics["assets"] else 0.0,
        "mean_row_f1": float(np.mean([asset_metrics["f1_score"] for asset_metrics in all_metrics["assets"].values()])) if all_metrics["assets"] else 0.0,
    }
    all_metrics["communication"] = {
        "raw_dataset_bytes": int(raw_data_bytes),
        "model_update_bytes": int(communication_bytes),
        "reduction_ratio": float(1 - (communication_bytes / raw_data_bytes)) if raw_data_bytes else 0.0,
    }
    all_metrics["aws_cost_estimate_usd_per_month"] = 11.58
    write_json(metrics_dir / "evaluation_summary.json", all_metrics)
    return all_metrics
