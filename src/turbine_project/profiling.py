from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AppConfig
from .utils import write_json


def profile_processed_dataset(config: AppConfig, max_assets: int | None = None) -> dict[str, object]:
    assets = sorted(config.dataset.processed_dir.glob("asset_id=*"))
    if max_assets is not None:
        assets = assets[:max_assets]

    report: dict[str, object] = {
        "feature_columns": config.dataset.feature_columns,
        "assets": {},
    }

    for asset_dir in assets:
        asset_id = asset_dir.name.split("=", maxsplit=1)[1]
        frame = pd.concat([pd.read_parquet(path) for path in sorted(asset_dir.glob("chunk_*.parquet"))], ignore_index=True)
        label_counts = pd.to_numeric(frame[config.dataset.label_column], errors="coerce").fillna(config.dataset.healthy_label).astype(int).value_counts().to_dict()
        feature_stats = {}
        for feature in config.dataset.feature_columns[:8]:
            feature_stats[feature] = {
                "mean": float(frame[feature].mean()),
                "std": float(frame[feature].std(ddof=0)),
                "min": float(frame[feature].min()),
                "max": float(frame[feature].max()),
            }
        report["assets"][asset_id] = {
            "rows": int(len(frame)),
            "label_distribution": {str(key): int(value) for key, value in label_counts.items()},
            "feature_stats": feature_stats,
        }

    path = config.dataset.reports_dir / "dataset_profile.json"
    write_json(path, report)
    return report
