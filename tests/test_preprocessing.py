from pathlib import Path

import pandas as pd

from turbine_project.config import DatasetConfig
from turbine_project.preprocessing import _clean_chunk


def test_clean_chunk_imputes_and_preserves_feature() -> None:
    config = DatasetConfig(
        csv_path=Path("sample.csv"),
        processed_dir=Path("data"),
        reports_dir=Path("reports"),
        chunk_size=2,
        feature_columns=["power_2_avg"],
        timestamp_column="time_stamp",
        asset_id_column="asset_id",
        split_column="train_test",
        label_column="status_type_id",
        split_train_value="train",
        split_test_value="prediction",
        healthy_label=0,
        anomaly_labels=[3, 4, 5],
        low_quantile=0.01,
        high_quantile=0.99,
        event_gap_minutes=30,
        min_required_columns=["time_stamp", "asset_id", "train_test", "status_type_id", "power_2_avg"],
    )
    frame = pd.DataFrame(
        {
            "time_stamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"],
            "asset_id": [1, 1],
            "train_test": ["train", "train"],
            "status_type_id": [0, 0],
            "power_2_avg": [1.0, None],
        }
    )
    cleaned, metrics = _clean_chunk(frame, config)
    assert cleaned["power_2_avg"].isna().sum() == 0
    assert metrics["missing_before"] == 1
