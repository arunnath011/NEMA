"""Feature engineering for the NEMA load forecasting model.

Reproduces every feature transform from the research notebook:
  - Cyclic calendar encoding (hour, day-of-week, month, day-of-year)
  - US / MA holiday indicators (+ day before/after)
  - Temperature-derived features (HDD, CDD, polynomial, sigmoidal)
  - Sequence creation (168 h lookback windows)
  - Lag / rolling-mean extraction for gradient-boosting input
"""

from __future__ import annotations

import logging

import holidays
import numpy as np
import pandas as pd

from nema_forecast.config import HORIZON, LAG_HOURS, LOOKBACK, ROLLING_WINDOWS

logger = logging.getLogger(__name__)

_US_HOLIDAYS = holidays.country_holidays("US", years=range(2017, 2027))
_MA_HOLIDAYS = holidays.country_holidays("US", subdiv="MA", years=range(2017, 2027))

BASE_TEMP_F = 65.0  # base temperature for HDD/CDD (°F)

# Features selected by PCMCI causal analysis in the notebook
CAUSAL_WEATHER_FEATURES = ["temp", "humidity", "wind_speed", "visibility"]

# Explicit RTLO lag features that survived the ablation study
RTLO_KEEP = frozenset(
    {
        "RTLO_lag4",
        "RTLO_mean24",
        "RTLO_lag48",
        "RTLO_lag168",
        "RTLO_mean168",
    }
)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def add_calendar_features(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    dt = pd.to_datetime(df[col])
    df = df.copy()

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# Holiday features
# ---------------------------------------------------------------------------


def add_holiday_features(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df[col]).dt.date
    df["is_us_holiday"] = dates.apply(lambda d: d in _US_HOLIDAYS).astype(int)
    return df


# ---------------------------------------------------------------------------
# Temperature / weather features
# ---------------------------------------------------------------------------


def add_temperature_features(df: pd.DataFrame, temp_col: str = "temp") -> pd.DataFrame:
    """HDD, CDD, polynomial, and sigmoidal temperature features."""
    df = df.copy()
    if temp_col not in df.columns:
        return df

    temp = df[temp_col]
    df["HDD"] = np.maximum(BASE_TEMP_F - temp, 0)
    df["CDD"] = np.maximum(temp - BASE_TEMP_F, 0)
    df["temp_sq"] = temp**2
    df["HDD_sq"] = df["HDD"] ** 2
    df["CDD_sq"] = df["CDD"] ** 2

    # Sigmoidal transforms — capture non-linear heating/cooling demand
    df["temp_heating_sigmoid"] = 1 - _sigmoid(temp, center=18, steepness=0.3)
    df["temp_cooling_sigmoid"] = _sigmoid(temp, center=22, steepness=0.3)
    df["temp_thermal_stress"] = df["temp_heating_sigmoid"] + df["temp_cooling_sigmoid"]

    return df


def _sigmoid(x: pd.Series, center: float, steepness: float = 0.3) -> pd.Series:
    result: pd.Series = 1 / (1 + np.exp(-steepness * (x - center)))
    return result


# ---------------------------------------------------------------------------
# Full feature-engineering pipeline
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature pipeline and select the final column set."""
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_temperature_features(df)

    keep = [
        "RTLO",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_us_holiday",
    ]
    extras = [
        "temp",
        "humidity",
        "wind_speed",
        "visibility",
        "clouds_all",
        "HDD",
        "CDD",
        "temp_sq",
        "HDD_sq",
        "CDD_sq",
        "temp_heating_sigmoid",
        "temp_cooling_sigmoid",
        "temp_thermal_stress",
    ]
    keep += [c for c in extras if c in df.columns]
    keep = [c for c in keep if c in df.columns]
    return df[["datetime"] + keep].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sequence / lag creation for CatBoost
# ---------------------------------------------------------------------------


def create_sequences(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Slide a lookback window over *df* to produce (X, Y) arrays.

    Returns
    -------
    X : ndarray of shape (N, lookback, F)
    Y : ndarray of shape (N, horizon)
    feature_cols : list of feature column names
    rtlo_idx : index of RTLO in *feature_cols*
    """
    feature_cols = [c for c in df.columns if c != "datetime"]
    rtlo_idx = feature_cols.index("RTLO")
    values = df[feature_cols].values
    n = len(values)

    x_list, y_list = [], []
    for t in range(lookback, n - horizon):
        x_list.append(values[t - lookback : t])
        y_list.append(values[t : t + horizon, rtlo_idx])
    return np.stack(x_list), np.stack(y_list), feature_cols, rtlo_idx


def extract_lag_features(
    X: np.ndarray,
    feature_names: list[str],
    rtlo_idx: int,
    lag_hours: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert 3-D sequence array into 2-D lag-feature matrix for CatBoost.

    Applies the same RTLO-feature whitelist from the notebook ablation study.
    """
    lag_hours = lag_hours or LAG_HOURS
    rolling_windows = rolling_windows or ROLLING_WINDOWS

    _n, length, _f = X.shape
    parts: list[np.ndarray] = []
    names: list[str] = []

    for lag in lag_hours:
        if lag > length:
            continue
        parts.append(X[:, length - lag, :])
        names.extend(f"{f}_lag{lag}" for f in feature_names)

    for win in rolling_windows:
        if win > length:
            continue
        parts.append(np.mean(X[:, -win:, :], axis=1))
        names.extend(f"{f}_mean{win}" for f in feature_names)

    all_features = np.hstack(parts)

    # Apply the RTLO whitelist (keep only the lags that help, drop the rest)
    keep_idx: list[int] = []
    keep_names: list[str] = []
    for i, name in enumerate(names):
        if name.startswith("RTLO_"):
            if name in RTLO_KEEP:
                keep_idx.append(i)
                keep_names.append(name)
        else:
            keep_idx.append(i)
            keep_names.append(name)

    return all_features[:, keep_idx], keep_names
