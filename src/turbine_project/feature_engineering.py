from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Real SCADA exports (e.g. the CARE-to-Compare dataset this project is
# built on) number each turbine's sensors locally per file/export - the
# same physical role (power output, wind speed, ...) can show up as
# "power_2_avg" in one file and "power_29_avg" in another. Hardcoding one
# set of IDs only works for the file that happened to use those IDs, so
# columns are matched by role + statistic instead, and each role's
# available columns (there can be several sensors of the same kind) are
# aggregated into one feature per role/statistic pair.
_ROLES = ("power", "wind_speed", "reactive_power", "sensor")
_STATS = ("avg", "max", "min", "std")
_ROLE_PATTERN = re.compile(r"^(" + "|".join(_ROLES) + r")_\d+(?:_(" + "|".join(_STATS) + r"))?$")

ENGINEERED_FEATURE_COLUMNS = [
    "power_avg",
    "power_max",
    "power_min",
    "power_std",
    "wind_speed_avg",
    "wind_speed_max",
    "wind_speed_min",
    "wind_speed_std",
    "reactive_power_avg",
    "reactive_power_max",
    "reactive_power_min",
    "reactive_power_std",
    "sensor_avg",
    "sensor_std",
    "power_wind_ratio",
    "wind_speed_spread",
    "reactive_power_spread",
    "power_band",
    "power_rolling_mean_3",
    "power_rolling_std_3",
    "power_delta_1",
]


def classify_columns(columns) -> dict[str, dict[str, list[str]]]:
    """Group raw SCADA columns by physical role and statistic variant.

    A bare column with no ``_avg``/``_max``/``_min``/``_std`` suffix (some
    sensors only ever report one scalar per timestamp) is treated as that
    role's "avg" reading.
    """
    roles: dict[str, dict[str, list[str]]] = {role: {stat: [] for stat in _STATS} for role in _ROLES}
    for column in columns:
        match = _ROLE_PATTERN.match(column)
        if not match:
            continue
        role = match.group(1)
        stat = match.group(2) or "avg"
        roles[role][stat].append(column)
    return roles


def available_roles(columns) -> set[str]:
    """Roles that have at least one matching column, for schema validation."""
    roles = classify_columns(columns)
    return {role for role, stats in roles.items() if stats["avg"]}


def add_engineered_features(frame: pd.DataFrame, asset_id_column: str) -> pd.DataFrame:
    engineered = frame.copy()
    eps = 1e-6
    roles = classify_columns(engineered.columns)

    role_stat_values: dict[str, dict[str, pd.Series]] = {}
    for role in _ROLES:
        avg_columns = roles[role]["avg"]
        if avg_columns:
            avg_series = engineered[avg_columns].mean(axis=1)
        else:
            LOGGER.warning("No '%s' columns found in this batch; defaulting that feature to 0.0", role)
            avg_series = pd.Series(0.0, index=engineered.index)
        stat_values = {"avg": avg_series}
        for stat in ("max", "min", "std"):
            columns = roles[role][stat]
            stat_values[stat] = engineered[columns].mean(axis=1) if columns else avg_series
        role_stat_values[role] = stat_values

    for role in _ROLES:
        for stat in _STATS:
            engineered[f"{role}_{stat}"] = role_stat_values[role][stat]

    engineered["power_wind_ratio"] = engineered["power_avg"] / (engineered["wind_speed_avg"].abs() + eps)
    engineered["wind_speed_spread"] = engineered["wind_speed_max"] - engineered["wind_speed_min"]
    engineered["reactive_power_spread"] = engineered["reactive_power_max"] - engineered["reactive_power_min"]
    engineered["power_band"] = engineered["power_max"] - engineered["power_min"]

    grouped = engineered.groupby(asset_id_column, group_keys=False)
    engineered["power_rolling_mean_3"] = grouped["power_avg"].transform(
        lambda series: series.rolling(window=3, min_periods=1).mean()
    )
    engineered["power_rolling_std_3"] = grouped["power_avg"].transform(
        lambda series: series.rolling(window=3, min_periods=1).std().fillna(0.0)
    )
    engineered["power_delta_1"] = grouped["power_avg"].transform(lambda series: series.diff().fillna(0.0))

    numeric_columns = engineered.select_dtypes(include=[np.number]).columns
    engineered[numeric_columns] = engineered[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return engineered
