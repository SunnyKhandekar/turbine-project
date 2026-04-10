from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from .config import DatasetConfig
from .feature_engineering import add_engineered_features
from .utils import ensure_directory, write_json

LOGGER = logging.getLogger(__name__)


def _clean_chunk(chunk: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    chunk = chunk.copy()
    chunk[config.timestamp_column] = pd.to_datetime(chunk[config.timestamp_column], errors="coerce", utc=True)
    chunk = chunk.dropna(subset=[config.timestamp_column, config.asset_id_column])

    numeric_columns = [column for column in config.feature_columns if column in chunk.columns]
    if not numeric_columns:
        raise ValueError("None of the configured feature columns exist in the incoming chunk.")

    base_numeric_columns = [
        column
        for column in chunk.columns
        if column not in {config.timestamp_column, config.asset_id_column, config.split_column}
    ]
    for column in base_numeric_columns:
        if column in {config.asset_id_column, config.label_column}:
            continue
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    missing_before = int(chunk[numeric_columns].isna().sum().sum())
    grouped = chunk.groupby(config.asset_id_column, group_keys=False)
    chunk[base_numeric_columns] = grouped[base_numeric_columns].transform(
        lambda frame: frame.ffill().bfill().interpolate(limit_direction="both")
    )
    chunk[base_numeric_columns] = chunk[base_numeric_columns].fillna(0.0)
    chunk = add_engineered_features(chunk, config.asset_id_column)

    low = chunk[numeric_columns].quantile(config.low_quantile)
    high = chunk[numeric_columns].quantile(config.high_quantile)
    chunk[numeric_columns] = chunk[numeric_columns].clip(lower=low, upper=high, axis=1)
    missing_after = int(chunk[numeric_columns].isna().sum().sum())

    return chunk.sort_values([config.asset_id_column, config.timestamp_column]), {
        "rows": int(len(chunk)),
        "missing_before": missing_before,
        "missing_after": missing_after,
    }


def preprocess_dataset(config: DatasetConfig, max_input_chunks: int | None = None) -> dict[str, object]:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.csv_path}")

    ensure_directory(config.processed_dir)
    ensure_directory(config.reports_dir)

    header = pd.read_csv(config.csv_path, nrows=0)
    missing_columns = sorted(set(config.min_required_columns) - set(header.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    asset_buffers: dict[str, list[pd.DataFrame]] = defaultdict(list)
    asset_rows: dict[str, int] = defaultdict(int)
    asset_chunk_counts: dict[str, int] = defaultdict(int)
    total_rows = 0
    processed_chunks = 0
    report = {
        "csv_path": str(config.csv_path),
        "chunk_size": config.chunk_size,
        "feature_columns": config.feature_columns,
        "assets": {},
        "processed_input_chunks": 0,
        "total_rows": 0,
    }

    def flush_asset(asset_id: str, force: bool = False) -> None:
        if not asset_buffers[asset_id]:
            return
        frame = pd.concat(asset_buffers[asset_id], ignore_index=True)
        while len(frame) >= config.chunk_size or (force and not frame.empty):
            to_write = frame.iloc[: config.chunk_size].copy()
            frame = frame.iloc[config.chunk_size :].copy()
            asset_dir = ensure_directory(config.processed_dir / f"asset_id={asset_id}")
            output_path = asset_dir / f"chunk_{asset_chunk_counts[asset_id]:05d}.parquet"
            to_write.to_parquet(output_path, index=False)
            asset_chunk_counts[asset_id] += 1
            LOGGER.info("Saved %s rows for asset %s to %s", len(to_write), asset_id, output_path)
        asset_buffers[asset_id] = [frame] if not frame.empty else []

    for chunk_index, raw_chunk in enumerate(pd.read_csv(config.csv_path, chunksize=config.chunk_size)):
        if max_input_chunks is not None and chunk_index >= max_input_chunks:
            break
        cleaned_chunk, metrics = _clean_chunk(raw_chunk, config)
        processed_chunks += 1
        total_rows += metrics["rows"]
        for asset_id, asset_frame in cleaned_chunk.groupby(config.asset_id_column):
            asset_id_str = str(asset_id)
            asset_buffers[asset_id_str].append(asset_frame)
            asset_rows[asset_id_str] += int(len(asset_frame))
            flush_asset(asset_id_str)

    for asset_id in list(asset_buffers):
        flush_asset(asset_id, force=True)

    report["processed_input_chunks"] = processed_chunks
    report["total_rows"] = total_rows
    report["assets"] = {
        asset_id: {"rows": asset_rows[asset_id], "chunks": asset_chunk_counts[asset_id]}
        for asset_id in sorted(asset_rows)
    }
    write_json(config.reports_dir / "preprocessing_report.json", report)
    return report
