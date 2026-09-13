from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import DatasetConfig
from .feature_engineering import (
    add_engineered_features,
    available_roles,
    build_sensor_categories,
    load_feature_description,
)
from .utils import ensure_directory, write_json

LOGGER = logging.getLogger(__name__)

# Physical signals a turbine model needs at minimum. Reactive power and
# generic "sensor" columns are useful but not fatal if a particular export
# omits them (add_engineered_features falls back to 0.0 for those with a
# logged warning); missing power or wind speed means the file can't
# support anomaly detection at all.
_REQUIRED_ROLES = {"power", "wind_speed"}
_DELIMITER_CANDIDATES = (",", ";", "\t", "|")
_METADATA_COLUMN_HINTS = ("time_stamp", "asset_id", "status_type_id", "train_test")


def detect_delimiter(csv_path: Path, configured: str = "auto") -> str:
    """Pick the field delimiter that actually splits the header correctly.

    Real-world SCADA exports (e.g. the CARE-to-Compare dataset) are often
    semicolon-separated; assuming comma silently parses every row as one
    giant column instead of failing loudly. A wide export (Farm B has 257
    columns, Farm C has 957) can make a fixed-size byte sample land mid-row
    or capture too few complete lines for statistical sniffing to work
    reliably, so this reads only the header line and scores each candidate
    delimiter by whether it actually produces the metadata columns every
    CARE-to-Compare file has - independent of row width.
    """
    if configured != "auto":
        return configured
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header_line = handle.readline()
    best_delimiter, best_score = ",", -1
    for delimiter in _DELIMITER_CANDIDATES:
        columns = header_line.rstrip("\r\n").split(delimiter)
        if len(columns) <= 1:
            continue
        score = sum(1 for hint in _METADATA_COLUMN_HINTS if hint in columns)
        if score > best_score:
            best_delimiter, best_score = delimiter, score
    if best_score <= 0:
        LOGGER.warning(
            "Could not confidently detect a delimiter for %s (no expected metadata "
            "columns found with any candidate); defaulting to ','",
            csv_path,
        )
    return best_delimiter


def _locate_feature_description(*candidate_dirs: Path) -> Path | None:
    """Find a feature_description.csv near the data, checking each dir in order.

    Every CARE-to-Compare farm ships one of these next to its data (either
    beside a single file, or one level up from a farm's ``datasets`` folder).
    """
    for directory in candidate_dirs:
        for candidate in directory.glob("*.csv"):
            if candidate.name.lower() == "feature_description.csv":
                return candidate
    return None


def _clean_chunk(
    chunk: pd.DataFrame, config: DatasetConfig, sensor_categories: dict[str, str] | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    chunk = chunk.copy()
    chunk[config.timestamp_column] = pd.to_datetime(chunk[config.timestamp_column], errors="coerce", utc=True)
    chunk = chunk.dropna(subset=[config.timestamp_column, config.asset_id_column])

    base_numeric_columns = [
        column
        for column in chunk.columns
        if column not in {config.timestamp_column, config.asset_id_column, config.split_column}
    ]
    for column in base_numeric_columns:
        if column in {config.asset_id_column, config.label_column}:
            continue
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    # Measured on the raw incoming sensor columns (before imputation), so
    # this reflects actual data quality rather than the derived feature set.
    missing_before = int(chunk[base_numeric_columns].isna().sum().sum())

    grouped = chunk.groupby(config.asset_id_column, group_keys=False)
    chunk[base_numeric_columns] = grouped[base_numeric_columns].transform(
        lambda frame: frame.ffill().bfill().interpolate(limit_direction="both")
    )
    chunk[base_numeric_columns] = chunk[base_numeric_columns].fillna(0.0)
    missing_after = int(chunk[base_numeric_columns].isna().sum().sum())

    # Engineered, role-based features are computed here (rather than
    # accepted as literal config.feature_columns from the raw file), so
    # they always exist regardless of which exact sensor IDs this file uses.
    chunk = add_engineered_features(chunk, config.asset_id_column, sensor_categories)
    numeric_columns = [column for column in config.feature_columns if column in chunk.columns]

    low = chunk[numeric_columns].quantile(config.low_quantile)
    high = chunk[numeric_columns].quantile(config.high_quantile)
    chunk[numeric_columns] = chunk[numeric_columns].clip(lower=low, upper=high, axis=1)
    chunk[numeric_columns] = chunk[numeric_columns].fillna(0.0)

    return chunk.sort_values([config.asset_id_column, config.timestamp_column]), {
        "rows": int(len(chunk)),
        "missing_before": missing_before,
        "missing_after": missing_after,
    }


def _validate_schema(csv_path: Path, config: DatasetConfig, delimiter: str) -> list[str]:
    """Return the header's columns after validating structure, or raise."""
    header = pd.read_csv(csv_path, nrows=0, sep=delimiter)
    missing_columns = sorted(set(config.min_required_columns) - set(header.columns))
    if missing_columns:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing_columns}")

    missing_roles = _REQUIRED_ROLES - available_roles(header.columns)
    if missing_roles:
        raise ValueError(
            f"{csv_path.name} has none of the required signal types: {sorted(missing_roles)}. "
            "Expected at least one column per role matching '<role>_<id>[_avg|_max|_min|_std]', "
            "e.g. 'power_2_avg' or 'wind_speed_3'."
        )
    return list(header.columns)


class _Ingester:
    """Accumulates cleaned rows per turbine across one or more source CSVs.

    A single farm's raw data is many small per-event files, not one combined
    CSV (e.g. 22 files for Wind Farm A). Each file can contain rows for
    turbines already seen in a previous file, so the per-asset output chunk
    counter must carry over across files instead of restarting at 0 - restart
    it and a second file silently overwrites the first file's chunk_00000.
    """

    def __init__(
        self,
        config: DatasetConfig,
        resume: bool = True,
        sensor_categories: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.sensor_categories = sensor_categories
        self.asset_buffers: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self.asset_rows: dict[str, int] = defaultdict(int)
        self.asset_chunk_counts: dict[str, int] = defaultdict(int)
        self.processed_chunks = 0
        self.total_rows = 0
        self.total_input_bytes = 0
        self.files_processed: list[str] = []
        self.files_skipped: dict[str, str] = {}
        if resume:
            self._resume_chunk_counts()

    def _resume_chunk_counts(self) -> None:
        if not self.config.processed_dir.exists():
            return
        for asset_dir in self.config.processed_dir.glob("asset_id=*"):
            asset_id = asset_dir.name.split("=", maxsplit=1)[1]
            existing = sorted(asset_dir.glob("chunk_*.parquet"))
            if existing:
                self.asset_chunk_counts[asset_id] = len(existing)

    def flush_asset(self, asset_id: str, force: bool = False) -> None:
        if not self.asset_buffers[asset_id]:
            return
        frame = pd.concat(self.asset_buffers[asset_id], ignore_index=True)
        while len(frame) >= self.config.chunk_size or (force and not frame.empty):
            to_write = frame.iloc[: self.config.chunk_size].copy()
            frame = frame.iloc[self.config.chunk_size :].copy()
            asset_dir = ensure_directory(self.config.processed_dir / f"asset_id={asset_id}")
            output_path = asset_dir / f"chunk_{self.asset_chunk_counts[asset_id]:05d}.parquet"
            to_write.to_parquet(output_path, index=False)
            self.asset_chunk_counts[asset_id] += 1
            LOGGER.info("Saved %s rows for asset %s to %s", len(to_write), asset_id, output_path)
        self.asset_buffers[asset_id] = [frame] if not frame.empty else []

    def ingest_file(self, csv_path: Path, max_input_chunks: int | None = None) -> None:
        delimiter = detect_delimiter(csv_path, self.config.csv_delimiter)
        try:
            _validate_schema(csv_path, self.config, delimiter)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", csv_path.name, exc)
            self.files_skipped[csv_path.name] = str(exc)
            return

        self.total_input_bytes += csv_path.stat().st_size
        for chunk_index, raw_chunk in enumerate(pd.read_csv(csv_path, chunksize=self.config.chunk_size, sep=delimiter)):
            if max_input_chunks is not None and chunk_index >= max_input_chunks:
                break
            cleaned_chunk, metrics = _clean_chunk(raw_chunk, self.config, self.sensor_categories)
            self.processed_chunks += 1
            self.total_rows += metrics["rows"]
            for asset_id, asset_frame in cleaned_chunk.groupby(self.config.asset_id_column):
                asset_id_str = str(asset_id)
                self.asset_buffers[asset_id_str].append(asset_frame)
                self.asset_rows[asset_id_str] += int(len(asset_frame))
                self.flush_asset(asset_id_str)
        self.files_processed.append(csv_path.name)

    def finalize(self) -> dict[str, object]:
        for asset_id in list(self.asset_buffers):
            self.flush_asset(asset_id, force=True)
        return {
            "processed_input_chunks": self.processed_chunks,
            "total_rows": self.total_rows,
            "total_input_bytes": self.total_input_bytes,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "assets": {
                asset_id: {"rows": self.asset_rows[asset_id], "chunks": self.asset_chunk_counts[asset_id]}
                for asset_id in sorted(set(self.asset_rows) | set(self.asset_chunk_counts))
            },
        }


def preprocess_dataset(config: DatasetConfig, max_input_chunks: int | None = None) -> dict[str, object]:
    if not config.csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.csv_path}")

    ensure_directory(config.processed_dir)
    ensure_directory(config.reports_dir)

    description_path = _locate_feature_description(config.csv_path.parent)
    sensor_categories = build_sensor_categories(load_feature_description(description_path) if description_path else None)

    ingester = _Ingester(config, resume=False, sensor_categories=sensor_categories)
    ingester.ingest_file(config.csv_path, max_input_chunks=max_input_chunks)
    if config.csv_path.name in ingester.files_skipped:
        raise ValueError(ingester.files_skipped[config.csv_path.name])

    report = {
        "csv_path": str(config.csv_path),
        "chunk_size": config.chunk_size,
        "feature_columns": config.feature_columns,
        "feature_description_path": str(description_path) if description_path else None,
        "sensor_categories": sensor_categories,
    }
    report.update(ingester.finalize())
    write_json(config.reports_dir / "preprocessing_report.json", report)
    return report


def preprocess_source_directory(
    config: DatasetConfig,
    source_dir: Path,
    max_input_files: int | None = None,
    max_input_chunks_per_file: int | None = None,
) -> dict[str, object]:
    """Ingest every CSV in a directory (e.g. one farm's ``datasets`` folder).

    Each file's rows accumulate into the same per-turbine parquet chunks
    (resuming existing chunk counters rather than overwriting them), so this
    can process a whole farm's many per-event files as one dataset, and can
    be re-run incrementally without destroying prior output.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Metadata files (feature_description.csv, event_info.csv) sometimes sit
    # alongside the per-event data files rather than strictly one level up -
    # exclude them by name so they're never mistaken for turbine data.
    _metadata_filenames = {"feature_description.csv", "event_info.csv"}
    csv_paths = sorted(path for path in source_dir.glob("*.csv") if path.name.lower() not in _metadata_filenames)
    if max_input_files is not None:
        csv_paths = csv_paths[:max_input_files]
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    ensure_directory(config.processed_dir)
    ensure_directory(config.reports_dir)

    # A farm's feature_description.csv sits either inside the "datasets"
    # folder itself or one level up (the layout CARE-to-Compare ships), so
    # check both rather than assuming one.
    description_path = _locate_feature_description(source_dir, source_dir.parent)
    if description_path:
        LOGGER.info("Using feature description metadata from %s", description_path)
    else:
        LOGGER.warning(
            "No feature_description.csv found near %s; generic sensors will be "
            "aggregated as one undifferentiated bucket instead of by physical meaning "
            "(temperature/rotational speed/electrical).",
            source_dir,
        )
    sensor_categories = build_sensor_categories(load_feature_description(description_path) if description_path else None)

    ingester = _Ingester(config, resume=True, sensor_categories=sensor_categories)
    for csv_path in csv_paths:
        LOGGER.info("Ingesting %s", csv_path)
        ingester.ingest_file(csv_path, max_input_chunks=max_input_chunks_per_file)

    report = {
        "source_dir": str(source_dir),
        "chunk_size": config.chunk_size,
        "feature_columns": config.feature_columns,
        "feature_description_path": str(description_path) if description_path else None,
        "sensor_categories": sensor_categories,
    }
    report.update(ingester.finalize())
    write_json(config.reports_dir / "preprocessing_report.json", report)
    return report
