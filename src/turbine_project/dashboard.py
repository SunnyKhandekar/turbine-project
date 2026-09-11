from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_output_path(root: Path, stored_path: str | None) -> Path | None:
    """Resolve a path recorded in a metrics report against this deployment.

    Reports may have been generated on a different machine/OS (e.g. a local
    Windows training run, later deployed to Streamlit Cloud on Linux), so a
    stored path can be absolute and simply not exist here. Recover the
    portion of the path under ``outputs/`` and re-root it locally instead of
    trusting the stored value directly.
    """
    if not stored_path:
        return None
    normalized = stored_path.replace("\\", "/")
    marker = "outputs/"
    index = normalized.find(marker)
    relative = normalized[index:] if index != -1 else Path(normalized).name
    candidate = (root / relative).resolve()
    return candidate if candidate.exists() else None


@st.cache_data(show_spinner=False)
def _load_predictions(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _portfolio_table(summary: dict) -> pd.DataFrame:
    rows = []
    for asset_id, metrics in summary["assets"].items():
        rows.append(
            {
                "Turbine": asset_id,
                "Rows": metrics["rows"],
                "Row F1": round(metrics["f1_score"], 3),
                "Event F1": round(metrics["event_f1"], 3),
                "Precision": round(metrics["precision"], 3),
                "Recall": round(metrics["recall"], 3),
                "Actual Events": metrics["actual_events"],
                "Predicted Events": metrics["predicted_events"],
                "Inference ms/row": round(metrics["latency_ms_per_row"], 3),
                "Turbine Correct": "Yes" if metrics.get("turbine_level_correct") == 1 else "No",
            }
        )
    return pd.DataFrame(rows).sort_values(["Event F1", "Row F1"], ascending=False)


def _top_drift_table(monitoring_path: Path) -> pd.DataFrame:
    monitoring = _read_json(monitoring_path)
    frame = pd.DataFrame(monitoring["feature_drift"]).sort_values("sigma_shift", ascending=False)
    return frame.head(8)


def _proxy_drift_table(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Approximate feature drift when no persisted monitoring snapshot exists.

    Compares predicted-normal vs predicted-anomalous rows within the same
    inference batch, standing in for the train-vs-inference comparison a
    real monitoring run would make.
    """
    rows = []
    normal = frame.loc[frame["anomaly_prediction"] == 0]
    anomalous = frame.loc[frame["anomaly_prediction"] == 1]
    if normal.empty or anomalous.empty:
        return pd.DataFrame(columns=["feature", "normal_mean", "anomalous_mean", "sigma_shift"])
    for feature in feature_names:
        normal_series = normal[feature].astype(float)
        normal_mean = float(normal_series.mean())
        normal_std = float(normal_series.std(ddof=0)) or 1e-6
        anomalous_mean = float(anomalous[feature].astype(float).mean())
        rows.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "anomalous_mean": anomalous_mean,
                "sigma_shift": abs(anomalous_mean - normal_mean) / normal_std,
            }
        )
    return pd.DataFrame(rows).sort_values("sigma_shift", ascending=False).head(8)


def _score_distribution_figure(frame: pd.DataFrame, asset_id: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(frame["anomaly_score"], bins=50, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"Turbine {asset_id}: anomaly score distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def _feature_scatter_figure(frame: pd.DataFrame, asset_id: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sample = frame.sample(min(len(frame), 5000), random_state=42)
    sns.scatterplot(
        data=sample,
        x="wind_speed_avg",
        y="power_avg",
        hue="anomaly_prediction",
        palette={0: "steelblue", 1: "darkorange"},
        alpha=0.6,
        s=20,
        ax=ax,
    )
    ax.set_title(f"Turbine {asset_id}: power vs wind-speed anomaly separation")
    fig.tight_layout()
    return fig


def _confusion_matrix_figure(metrics: dict, asset_id: str) -> plt.Figure | None:
    required = {"tn", "fp", "fn", "tp"}
    if not required.issubset(metrics):
        return None
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"], ax=ax)
    ax.set_title(f"Turbine {asset_id}: confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def run_dashboard(base_dir: str | Path = ".") -> None:
    root = Path(base_dir).resolve()
    metrics_path = root / "outputs" / "metrics" / "evaluation_summary.json"
    profile_path = root / "outputs" / "reports" / "dataset_profile.json"
    if not metrics_path.exists():
        st.error(f"Missing evaluation summary at {metrics_path}")
        return

    summary = _read_json(metrics_path)
    profile = _read_json(profile_path) if profile_path.exists() else {}
    portfolio = summary.get("portfolio_summary", {})
    comms = summary.get("communication", {})

    st.set_page_config(page_title="Wind Turbine Anomaly Platform", layout="wide")
    st.title("Wind Turbine Anomaly Detection Platform")
    st.caption("Executive dashboard for local production-style validation, turbine operations review, and interview/demo walkthroughs.")

    hero1, hero2, hero3, hero4 = st.columns(4)
    hero1.metric("Assets Evaluated", portfolio.get("assets_evaluated", 0))
    hero2.metric("Turbine Accuracy", _format_percent(portfolio.get("turbine_accuracy")))
    hero3.metric("Mean Event F1", f"{portfolio.get('mean_event_f1', 0.0):.3f}")
    hero4.metric("Communication Reduction", _format_percent(comms.get("reduction_ratio")))

    st.markdown("### Executive Takeaways")
    takeaway_col1, takeaway_col2 = st.columns([2, 1])
    with takeaway_col1:
        st.markdown(
            "\n".join(
                [
                    "- The system is optimized to catch fault events at turbine level while keeping raw SCADA data local.",
                    "- A multivariate `IsolationForest` detects deviations using power, wind-speed, and reactive-power behavior.",
                    "- Tree-weighted FedAvg-style aggregation provides a manager-safe federated story without overclaiming unsupported tree averaging.",
                    "- Monitoring outputs highlight drift so operators can decide when retraining is needed.",
                ]
            )
        )
    with takeaway_col2:
        st.info(
            f"Estimated cloud-equivalent monthly footprint: ${summary.get('aws_cost_estimate_usd_per_month', 0):.2f}\n\n"
            f"Raw dataset: {comms.get('raw_dataset_bytes', 0):,} bytes\n\n"
            f"Model updates: {comms.get('model_update_bytes', 0):,} bytes"
        )

    st.markdown("### Portfolio Overview")
    portfolio_table = _portfolio_table(summary)
    st.dataframe(portfolio_table, use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.bar_chart(portfolio_table.set_index("Turbine")[["Event F1", "Row F1"]], use_container_width=True)
    with chart_col2:
        event_frame = portfolio_table.set_index("Turbine")[["Actual Events", "Predicted Events"]]
        st.bar_chart(event_frame, use_container_width=True)

    st.markdown("### Turbine Drilldown")
    asset_id = st.selectbox("Select turbine", options=sorted(summary["assets"].keys()))
    asset_metrics = summary["assets"][asset_id]
    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
    detail_col1.metric("Row F1", f"{asset_metrics['f1_score']:.3f}")
    detail_col2.metric("Event F1", f"{asset_metrics['event_f1']:.3f}")
    detail_col3.metric("Recall", f"{asset_metrics['recall']:.3f}")
    detail_col4.metric("Latency per row", f"{asset_metrics['latency_ms_per_row']:.3f} ms")

    st.json(
        {
            "rows": asset_metrics["rows"],
            "actual_events": asset_metrics["actual_events"],
            "predicted_events": asset_metrics["predicted_events"],
            "precision": asset_metrics["precision"],
            "recall": asset_metrics["recall"],
            "average_precision": asset_metrics["average_precision"],
            "roc_auc": asset_metrics["roc_auc"],
            "turbine_level_correct": asset_metrics["turbine_level_correct"],
            "prediction_path": asset_metrics["prediction_path"],
        }
    )

    prediction_path = _resolve_output_path(root, asset_metrics.get("prediction_path")) or (
        root / "outputs" / "predictions" / f"asset_{asset_id}_predictions.parquet"
    )
    prediction_frame = None
    if prediction_path.exists():
        try:
            prediction_frame = _load_predictions(str(prediction_path))
        except Exception as exc:  # noqa: BLE001 - surface as a dashboard notice, never crash the page
            st.warning(f"Could not load predictions for turbine {asset_id}: {exc}")

    image_col1, image_col2 = st.columns(2)
    if prediction_frame is not None and {"wind_speed_avg", "power_avg", "anomaly_prediction"}.issubset(prediction_frame.columns):
        image_col1.pyplot(_feature_scatter_figure(prediction_frame, asset_id), use_container_width=True)
    if prediction_frame is not None and "anomaly_score" in prediction_frame.columns:
        image_col2.pyplot(_score_distribution_figure(prediction_frame, asset_id), use_container_width=True)
    if prediction_frame is None:
        st.info(f"Prediction data for turbine {asset_id} is not bundled with this deployment, so the score/feature plots are unavailable.")

    confusion_fig = _confusion_matrix_figure(asset_metrics, asset_id)
    if confusion_fig is not None:
        st.pyplot(confusion_fig, use_container_width=False)

    st.markdown("### Drift and Data Quality")
    monitoring_path = _resolve_output_path(root, asset_metrics.get("monitoring_path"))
    if monitoring_path is not None:
        drift = _top_drift_table(monitoring_path)
        drift_caption = "Feature drift: training baseline vs. inference batch."
    elif prediction_frame is not None and {"anomaly_prediction", *summary["feature_names"]}.issubset(prediction_frame.columns):
        drift = _proxy_drift_table(prediction_frame, summary["feature_names"])
        drift_caption = "No persisted monitoring snapshot was bundled with this deployment, so drift is approximated live from predicted-normal vs. predicted-anomalous rows in this batch."
    else:
        drift = pd.DataFrame()
        drift_caption = None

    if not drift.empty:
        st.caption(drift_caption)
        drift_col1, drift_col2 = st.columns([1.4, 1])
        with drift_col1:
            st.dataframe(drift, use_container_width=True, hide_index=True)
        with drift_col2:
            st.bar_chart(drift.set_index("feature")[["sigma_shift"]], use_container_width=True)
    else:
        st.info(f"No drift data is available for turbine {asset_id}.")

    if profile:
        st.markdown("### Dataset Context")
        asset_profile = profile.get("assets", {}).get(asset_id)
        if asset_profile:
            st.write("Observed label distribution and key feature ranges from processed data.")
            profile_col1, profile_col2 = st.columns(2)
            with profile_col1:
                label_frame = pd.DataFrame(
                    [{"status_type_id": key, "count": value} for key, value in asset_profile["label_distribution"].items()]
                )
                st.dataframe(label_frame, use_container_width=True, hide_index=True)
            with profile_col2:
                stats_frame = pd.DataFrame(asset_profile["feature_stats"]).T.reset_index().rename(columns={"index": "feature"})
                st.dataframe(stats_frame, use_container_width=True, hide_index=True)
