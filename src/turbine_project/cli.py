from __future__ import annotations

import argparse

from .config import AppConfig, load_config
from .preprocessing import preprocess_dataset
from .profiling import profile_processed_dataset
from .training import aggregate_local_models, evaluate_global_model, train_local_models
from .utils import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wind turbine federated anomaly detection pipeline")
    parser.add_argument("command", choices=["preprocess", "profile", "train", "aggregate", "evaluate", "run-all"])
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file")
    parser.add_argument("--max-input-chunks", type=int, default=None)
    parser.add_argument("--max-assets", type=int, default=None)
    return parser


def _run(command: str, config: AppConfig, max_input_chunks: int | None, max_assets: int | None) -> None:
    if command == "preprocess":
        preprocess_dataset(config.dataset, max_input_chunks=max_input_chunks)
    elif command == "profile":
        profile_processed_dataset(config, max_assets=max_assets)
    elif command == "train":
        train_local_models(config, max_assets=max_assets)
    elif command == "aggregate":
        aggregate_local_models(config)
    elif command == "evaluate":
        evaluate_global_model(config, max_assets=max_assets)
    elif command == "run-all":
        preprocess_dataset(config.dataset, max_input_chunks=max_input_chunks)
        profile_processed_dataset(config, max_assets=max_assets)
        train_local_models(config, max_assets=max_assets)
        aggregate_local_models(config)
        evaluate_global_model(config, max_assets=max_assets)
    else:
        raise ValueError(f"Unsupported command: {command}")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _run(args.command, config, args.max_input_chunks, args.max_assets)


if __name__ == "__main__":
    main()
