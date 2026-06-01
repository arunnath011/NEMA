"""Data cleaning, imputation, and merging for the NEMA pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from nema_forecast.config import DATA_CACHE_DIR, IMPUTATION_COLS, TRAIN_CUTOFF

logger = logging.getLogger(__name__)

# Columns that are >50 % missing in the original dataset — always drop
HIGH_MISSING_COLS = frozenset(
    {
        "sea_level",
        "grnd_level",
        "DA_Ancillary_Service_Cost",
        "Inventory_Energy_Program_Cost",
        "Price_Responsive_Demand_Cost",
        "snow_3h",
        "rain_3h",
        "snow_1h",
        "rain_1h",
        "wind_gust",
        "Real-Time_Demand_Reduction_Cost",
    }
)


def merge_load_weather(load_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join load data with weather on ``datetime``."""
    load_df = load_df.copy()
    weather_df = weather_df.copy()
    load_df["datetime"] = pd.to_datetime(load_df["datetime"])
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    # Round both to the nearest hour for a clean join
    load_df["datetime"] = load_df["datetime"].dt.floor("h")
    weather_df["datetime"] = weather_df["datetime"].dt.floor("h")
    weather_df = weather_df.drop_duplicates(subset=["datetime"], keep="first")

    merged = load_df.merge(weather_df, on="datetime", how="left")
    return merged.sort_values("datetime").reset_index(drop=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-missing columns and standardise names."""
    to_drop = [c for c in HIGH_MISSING_COLS if c in df.columns]
    if to_drop:
        logger.info("Dropping %d high-missing columns: %s", len(to_drop), to_drop)
        df = df.drop(columns=to_drop)
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def compute_imputation_stats(train_df: pd.DataFrame) -> dict[str, float]:
    """Compute column medians from training data only."""
    stats: dict[str, float] = {}
    for col in IMPUTATION_COLS:
        if col in train_df.columns and train_df[col].dtype in ("float64", "int64", "float32"):
            stats[col] = float(train_df[col].median())
    return stats


def save_imputation_stats(stats: dict[str, float], path: Path | None = None) -> Path:
    path = path or DATA_CACHE_DIR / "imputation_stats.json"
    path.write_text(json.dumps(stats, indent=2))
    logger.info("Saved imputation stats → %s", path)
    return path


def load_imputation_stats(path: Path | None = None) -> dict[str, float]:
    path = path or DATA_CACHE_DIR / "imputation_stats.json"
    stats: dict[str, float] = json.loads(path.read_text())
    return stats


def apply_imputation(df: pd.DataFrame, stats: dict[str, float]) -> pd.DataFrame:
    """Fill NaNs using pre-computed training medians, then ffill/bfill residuals."""
    df = df.copy()
    for col, val in stats.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    weather_cols = [c for c in IMPUTATION_COLS if c in df.columns]
    if weather_cols:
        df[weather_cols] = df[weather_cols].ffill().bfill()
    return df


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
    cutoff: str = TRAIN_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train / test by timestamp."""
    cutoff_dt = pd.to_datetime(cutoff)
    train = df[df["datetime"] <= cutoff_dt].copy()
    test = df[df["datetime"] > cutoff_dt].copy()
    return train, test
