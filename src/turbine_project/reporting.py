from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def write_text_report(
    evaluation_summary: dict[str, Any],
    output_path: Path,
    farm_name: str,
    preprocessing_report: dict[str, Any] | None = None,
) -> None:
    """Render a plain-text summary of a pipeline run for one farm.

    Intended as a human-readable companion to the JSON metrics - something
    that can be opened, diffed between runs, or shared without a dashboard.
    """
    lines: list[str] = []
    lines.append(f"Wind Turbine Anomaly Detection - {farm_name} Report")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")

    if preprocessing_report is not None:
        lines.append("=== Data Ingestion ===")
        files_processed = preprocessing_report.get("files_processed", [])
        files_skipped = preprocessing_report.get("files_skipped", {})
        lines.append(f"Source files processed: {len(files_processed)}")
        if files_skipped:
            lines.append(f"Source files skipped ({len(files_skipped)}):")
            for name, reason in files_skipped.items():
                lines.append(f"  - {name}: {reason}")
        lines.append(f"Total rows ingested: {preprocessing_report.get('total_rows', 0):,}")
        lines.append(f"Turbines found: {len(preprocessing_report.get('assets', {}))}")
        lines.append("")

    portfolio = evaluation_summary.get("portfolio_summary", {})
    comms = evaluation_summary.get("communication", {})
    lines.append("=== Portfolio Summary ===")
    lines.append(f"Assets evaluated: {portfolio.get('assets_evaluated', 0)}")
    lines.append(f"Turbine-level accuracy: {_fmt_pct(portfolio.get('turbine_accuracy'))}")
    lines.append(f"Mean event F1: {portfolio.get('mean_event_f1', 0.0):.3f}")
    lines.append(f"Mean row F1: {portfolio.get('mean_row_f1', 0.0):.3f}")
    lines.append(f"Communication reduction: {_fmt_pct(comms.get('reduction_ratio'))}")
    lines.append(f"Estimated cloud-equivalent monthly cost: ${evaluation_summary.get('aws_cost_estimate_usd_per_month', 0):.2f}")
    lines.append("")

    lines.append("=== Per-Turbine Metrics ===")
    header = (
        f"{'Turbine':<10}{'Rows':>8}{'Row F1':>9}{'Event F1':>10}{'Precision':>11}"
        f"{'Recall':>9}{'Events(A/P)':>13}{'Ambig.Excl.':>13}{'Mode':>20}{'Correct':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for asset_id, metrics in sorted(evaluation_summary.get("assets", {}).items()):
        correct = metrics.get("turbine_level_correct")
        correct_str = "Yes" if correct == 1 else ("No" if correct == 0 else "N/A")
        events = f"{metrics.get('actual_events', 0)}/{metrics.get('predicted_events', 0)}"
        lines.append(
            f"{asset_id:<10}{metrics.get('rows', 0):>8}{metrics.get('f1_score', 0.0):>9.3f}"
            f"{metrics.get('event_f1', 0.0):>10.3f}{metrics.get('precision', 0.0):>11.3f}"
            f"{metrics.get('recall', 0.0):>9.3f}{events:>13}"
            f"{metrics.get('ambiguous_rows_excluded', 0):>13}"
            f"{metrics.get('evaluation_mode', 'n/a'):>20}{correct_str:>9}"
        )
    lines.append("")

    lines.append("=== Notes ===")
    lines.append(
        "- 'Ambig.Excl.' counts rows with a status_type_id that is neither a "
        "confirmed healthy nor confirmed fault code (e.g. Derated, Idling, "
        "Other) - these are excluded from accuracy metrics rather than "
        "guessed either way."
    )
    lines.append(
        "- 'Mode' is 'observed_labels' when this turbine's prediction window "
        "contains confirmed ground truth, or 'synthetic_injection' when it "
        "doesn't (metrics are then estimated from synthetically injected "
        "anomalies rather than real labels)."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
