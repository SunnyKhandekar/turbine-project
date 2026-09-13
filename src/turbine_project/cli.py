from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig, load_config
from .preprocessing import preprocess_dataset, preprocess_source_directory
from .profiling import profile_processed_dataset
from .reporting import write_text_report
from .training import aggregate_local_models, evaluate_global_model, train_local_models
from .utils import read_json, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wind turbine federated anomaly detection pipeline")
    parser.add_argument(
        "command",
        choices=["preprocess", "profile", "train", "aggregate", "evaluate", "report", "run-all"],
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file")
    parser.add_argument("--max-input-chunks", type=int, default=None)
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument(
        "--source-dir",
        default=None,
        help=(
            "Directory of CSV files to ingest instead of the single config.dataset.csv_path "
            "(e.g. one farm's 'datasets' folder with many per-event files). Applies to "
            "'preprocess' and 'run-all'."
        ),
    )
    parser.add_argument(
        "--max-input-files",
        type=int,
        default=None,
        help="With --source-dir, limit how many files in the directory are ingested.",
    )
    parser.add_argument(
        "--report-txt",
        default=None,
        help="With 'report' or 'run-all', write a plain-text summary to this path.",
    )
    parser.add_argument(
        "--farm-name",
        default="Farm",
        help="Label used in the text report (e.g. 'Wind Farm A').",
    )
    return parser


def _run(command: str, config: AppConfig, args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir) if args.source_dir else None

    if command == "preprocess":
        if source_dir is not None:
            preprocess_source_directory(config.dataset, source_dir, max_input_files=args.max_input_files)
        else:
            preprocess_dataset(config.dataset, max_input_chunks=args.max_input_chunks)
    elif command == "profile":
        profile_processed_dataset(config, max_assets=args.max_assets)
    elif command == "train":
        train_local_models(config, max_assets=args.max_assets)
    elif command == "aggregate":
        aggregate_local_models(config)
    elif command == "evaluate":
        evaluate_global_model(config, max_assets=args.max_assets)
    elif command == "report":
        _write_report(config, args)
    elif command == "run-all":
        if source_dir is not None:
            preprocess_source_directory(config.dataset, source_dir, max_input_files=args.max_input_files)
        else:
            preprocess_dataset(config.dataset, max_input_chunks=args.max_input_chunks)
        profile_processed_dataset(config, max_assets=args.max_assets)
        train_local_models(config, max_assets=args.max_assets)
        aggregate_local_models(config)
        evaluate_global_model(config, max_assets=args.max_assets)
        if args.report_txt:
            _write_report(config, args)
    else:
        raise ValueError(f"Unsupported command: {command}")


def _write_report(config: AppConfig, args: argparse.Namespace) -> None:
    if not args.report_txt:
        raise ValueError("--report-txt is required for the 'report' command")
    evaluation_summary = read_json(config.inference.metrics_dir / "evaluation_summary.json")
    preprocessing_report_path = config.dataset.reports_dir / "preprocessing_report.json"
    preprocessing_report = read_json(preprocessing_report_path) if preprocessing_report_path.exists() else None
    write_text_report(evaluation_summary, Path(args.report_txt), args.farm_name, preprocessing_report)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _run(args.command, config, args)


if __name__ == "__main__":
    main()
