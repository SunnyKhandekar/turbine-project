from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "power_2_avg",
    "power_2_std",
    "power_5_avg",
    "power_6_avg",
    "wind_speed_235_avg",
    "wind_speed_236_avg",
    "reactive_power_119_avg",
    "reactive_power_120_avg",
    "sensor_41_avg",
    "sensor_42_avg",
]


def add_engineered_features(frame: pd.DataFrame, asset_id_column: str) -> pd.DataFrame:
    engineered = frame.copy()
    eps = 1e-6
    for column in BASE_COLUMNS:
        if column not in engineered.columns:
            engineered[column] = 0.0
    engineered["power_wind_ratio"] = engineered["power_2_avg"] / (engineered["wind_speed_236_avg"].abs() + eps)
    engineered["wind_speed_delta"] = engineered["wind_speed_236_avg"] - engineered["wind_speed_235_avg"]
    engineered["reactive_power_gap"] = engineered["reactive_power_120_avg"] - engineered["reactive_power_119_avg"]
    engineered["power_band"] = engineered["power_5_avg"] - engineered["power_6_avg"]

    grouped = engineered.groupby(asset_id_column, group_keys=False)
    engineered["power_rolling_mean_3"] = grouped["power_2_avg"].transform(
        lambda series: series.rolling(window=3, min_periods=1).mean()
    )
    engineered["power_rolling_std_3"] = grouped["power_2_avg"].transform(
        lambda series: series.rolling(window=3, min_periods=1).std().fillna(0.0)
    )
    engineered["power_delta_1"] = grouped["power_2_avg"].transform(lambda series: series.diff().fillna(0.0))

    numeric_columns = engineered.select_dtypes(include=[np.number]).columns
    engineered[numeric_columns] = engineered[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return engineered
