from __future__ import annotations

import logging
import re
from pathlib import Path

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
_ROLE_PATTERN = re.compile(r"^(" + "|".join(_ROLES) + r")_(\d+)(?:_(" + "|".join(_STATS) + r"))?$")

# "sensor_N" columns carry no physical meaning in their name (unlike
# power_*/wind_speed_*/reactive_power_*, which are already self-describing),
# so lumping every one of them into a single generic average is fragile: a
# temperature reading (~tens of degrees) gets averaged alongside a cumulative
# energy meter (~hundreds of thousands of Wh), and the huge-magnitude column
# silently dominates. When a feature_description.csv is available (every
# CARE-to-Compare farm ships one), it tells us what each sensor actually
# measures, so genuinely distinct signals (rotational speed, temperature,
# electrical) can be aggregated separately instead of blended together.
SEMANTIC_CATEGORIES = ("temperature", "rotational_speed", "electrical", "angle")
_ALL_GROUPS = _ROLES + SEMANTIC_CATEGORIES

_TEMPERATURE_KEYWORDS = ("temperature",)
_ROTATIONAL_SPEED_KEYWORDS = ("rpm",)
_ELECTRICAL_KEYWORDS = ("current", "voltage", "frequency", "phase displacement")

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
    "temperature_avg",
    "temperature_std",
    "rotational_speed_avg",
    "rotational_speed_std",
    "electrical_avg",
    "electrical_std",
    "angle_avg",
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


def load_feature_description(path: Path) -> pd.DataFrame | None:
    """Load a CARE-to-Compare-style feature_description.csv, if present.

    Returns None (rather than raising) when the file is missing or doesn't
    look like the expected format, so callers can fall back to purely
    name-pattern-based feature engineering.
    """
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, this is optional metadata
        LOGGER.warning("Could not read feature description file %s: %s", path, exc)
        return None
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "sensor_name" not in frame.columns or "description" not in frame.columns:
        LOGGER.warning("Feature description file %s is missing expected columns; ignoring", path)
        return None
    for boolean_column in ("is_angle", "is_counter"):
        if boolean_column in frame.columns:
            frame[boolean_column] = frame[boolean_column].astype(str).str.strip().str.lower().eq("true")
    return frame


def build_sensor_categories(feature_description: pd.DataFrame | None) -> dict[str, str]:
    """Map generic "sensor_N" ids to a semantic category from their description.

    Only re-buckets the generic "sensor_*" role - power/wind_speed/reactive_power
    columns already have clear names and keep their own dedicated role.
    """
    if feature_description is None:
        return {}
    categories: dict[str, str] = {}
    for _, row in feature_description.iterrows():
        sensor_name = str(row.get("sensor_name", ""))
        match = re.match(r"^sensor_(\d+)$", sensor_name)
        if not match:
            continue
        sensor_id = match.group(1)
        description = str(row.get("description", "")).lower()
        if bool(row.get("is_angle", False)):
            categories[sensor_id] = "angle"
        elif any(keyword in description for keyword in _TEMPERATURE_KEYWORDS):
            categories[sensor_id] = "temperature"
        elif any(keyword in description for keyword in _ROTATIONAL_SPEED_KEYWORDS):
            categories[sensor_id] = "rotational_speed"
        elif any(keyword in description for keyword in _ELECTRICAL_KEYWORDS):
            categories[sensor_id] = "electrical"
        # Anything else (including the large cumulative energy-meter columns,
        # which this dataset's own is_counter flag doesn't reliably mark)
        # stays in the generic "sensor" bucket - standardization below still
        # keeps it from dominating that bucket's average.
    return categories


def classify_columns(columns, sensor_categories: dict[str, str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Group raw SCADA columns by physical role/category and statistic variant.

    A bare column with no ``_avg``/``_max``/``_min``/``_std`` suffix (some
    sensors only ever report one scalar per timestamp) is treated as that
    group's "avg" reading.
    """
    groups = set(_ROLES) | set(sensor_categories.values()) if sensor_categories else set(_ROLES)
    groups |= set(SEMANTIC_CATEGORIES)
    buckets: dict[str, dict[str, list[str]]] = {group: {stat: [] for stat in _STATS} for group in groups}
    for column in columns:
        match = _ROLE_PATTERN.match(column)
        if not match:
            continue
        role, sensor_id, stat = match.group(1), match.group(2), match.group(3) or "avg"
        if role == "sensor" and sensor_categories and sensor_id in sensor_categories:
            role = sensor_categories[sensor_id]
        buckets[role][stat].append(column)
    return buckets


def available_roles(columns) -> set[str]:
    """Roles that have at least one matching column, for schema validation."""
    roles = classify_columns(columns)
    return {role for role, stats in roles.items() if stats["avg"]}


def _standardize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Z-score each column before it's blended into a group average.

    Without this, a single huge-magnitude column (e.g. a cumulative energy
    meter in the tens of thousands) silently dominates a group's mean and
    drowns out genuinely informative, smaller-magnitude signals (e.g.
    rotational speed) aggregated in the same bucket. Standardized per input
    chunk, matching this pipeline's existing per-chunk statistics (e.g. the
    quantile clipping in preprocessing._clean_chunk) rather than a global
    pass over the whole dataset.
    """
    subset = frame[columns].astype(float)
    std = subset.std(ddof=0).replace(0.0, 1.0)
    return (subset - subset.mean()) / std


def add_engineered_features(
    frame: pd.DataFrame,
    asset_id_column: str,
    sensor_categories: dict[str, str] | None = None,
) -> pd.DataFrame:
    engineered = frame.copy()
    eps = 1e-6
    roles = classify_columns(engineered.columns, sensor_categories)

    group_stat_values: dict[str, dict[str, pd.Series]] = {}
    for group in _ALL_GROUPS:
        avg_columns = roles[group]["avg"]
        if avg_columns:
            avg_series = _standardize(engineered, avg_columns).mean(axis=1)
        else:
            LOGGER.warning("No '%s' columns found in this batch; defaulting that feature to 0.0", group)
            avg_series = pd.Series(0.0, index=engineered.index)
        stat_values = {"avg": avg_series}
        for stat in ("max", "min", "std"):
            columns = roles[group][stat]
            stat_values[stat] = _standardize(engineered, columns).mean(axis=1) if columns else avg_series
        group_stat_values[group] = stat_values

    for group in _ALL_GROUPS:
        for stat in _STATS:
            engineered[f"{group}_{stat}"] = group_stat_values[group][stat]

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
