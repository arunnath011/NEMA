"""Inference — direct multi-horizon Beacon forecasts (augmented with target-hour features).

Each horizon h=1..24 has its own CatBoost model. The feature vector is the 168 h window lag
features (load history) **plus** the exogenous features at the target hour t+h — calendar
(exactly known) and weather (the forecast). That lets the day-ahead models use tomorrow's
forecasted temperature, which dominates load 24 h out.

Paths:
  * ``predict_next_24h``        — forward 24 h forecast, using the forecast weather for t+h.
  * ``predict_hindcast``        — 1-hour-ahead rolling hindcast (the "1 h-ahead" dashboard number).
  * ``predict_dayahead_hindcast`` — 24-hour-ahead rolling hindcast (fair vs ISO's day-ahead).

Weather comes from Open-Meteo (free, keyless) — the same source used for training — for both
historical hindcasts (archive) and the forward forecast, so target-hour weather features are
always populated. If weather is missing it falls back to training medians.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from nema_forecast.config import HORIZON, IMPUTATION_COLS, LOOKBACK, MODELS_DIR
from nema_forecast.data.preprocessing import apply_imputation, load_imputation_stats
from nema_forecast.features.engineering import (
    engineer_features,
    extract_lag_features,
    target_exog_indices,
)

logger = logging.getLogger(__name__)

_EMPTY = ["datetime", "actual", "forecast_mw"]


def _ensure_weather_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing weather columns as NaN so feature engineering yields the full schema."""
    df = df.copy()
    for col in IMPUTATION_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_model(path: Path | None = None) -> CatBoostRegressor:
    path = path or MODELS_DIR / "catboost_model.cbm"
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_horizon_models(horizon: int = HORIZON) -> list[CatBoostRegressor] | None:
    """Load per-horizon models ``catboost_h01..hNN.cbm``; None if any are missing."""
    paths = [MODELS_DIR / f"catboost_h{h:02d}.cbm" for h in range(1, horizon + 1)]
    if not all(p.exists() for p in paths):
        return None
    models = []
    for p in paths:
        m = CatBoostRegressor()
        m.load_model(str(p))
        models.append(m)
    return models


def _engineer(
    recent_load: pd.DataFrame,
    recent_weather: pd.DataFrame | None,
    future_dates: pd.DatetimeIndex | None = None,
):
    """Clean + impute + engineer features. Optionally append future rows (RTLO placeholder,
    weather from *recent_weather*) so target-hour exogenous features exist for forecasting.

    Returns ``(values, feature_cols, rtlo_idx, datetimes)``.
    """
    stats = load_imputation_stats(MODELS_DIR / "imputation_stats.json")
    load = recent_load[["datetime", "RTLO"]].copy()
    load["datetime"] = pd.to_datetime(load["datetime"]).dt.floor("h")
    load = load.dropna(subset=["RTLO"]).drop_duplicates("datetime").sort_values("datetime")

    full = load
    if future_dates is not None and len(future_dates) > 0:
        future = pd.DataFrame({"datetime": future_dates, "RTLO": load["RTLO"].iloc[-1]})
        full = pd.concat([load, future], ignore_index=True)

    if recent_weather is not None and not recent_weather.empty:
        wx = recent_weather.copy()
        wx["datetime"] = pd.to_datetime(wx["datetime"]).dt.floor("h")
        wx = wx.drop_duplicates("datetime")
        full = full.merge(wx, on="datetime", how="left")
    full = apply_imputation(_ensure_weather_cols(full), stats)

    feat = engineer_features(full)
    feature_cols = [c for c in feat.columns if c != "datetime"]
    values = feat[feature_cols].values.astype(float)
    dts = pd.to_datetime(feat["datetime"]).to_numpy()
    return values, feature_cols, feature_cols.index("RTLO"), dts


def _augment(x_gb: np.ndarray, exog: np.ndarray) -> np.ndarray:
    return np.hstack([x_gb, exog])


def _hindcast(
    values: np.ndarray,
    feature_cols: list[str],
    rtlo_idx: int,
    dts: np.ndarray,
    model: CatBoostRegressor,
    hi: int,
    max_hours: int | None,
) -> pd.DataFrame:
    """Rolling hindcast for horizon index *hi* (0-based): predict each hour j from the window
    ending at j-hi-1 plus the exogenous features at j. Returns ``[datetime, actual, forecast_mw]``."""
    exog_idx = target_exog_indices(feature_cols)
    n = len(values)
    start = hi + LOOKBACK
    if max_hours is not None:
        start = max(start, n - max_hours)
    targets = list(range(start, n))
    if not targets:
        return pd.DataFrame(columns=_EMPTY)

    windows = np.stack([values[j - hi - LOOKBACK : j - hi] for j in targets])
    x_gb, _ = extract_lag_features(windows, feature_cols, rtlo_idx)
    exog = np.stack([values[j, exog_idx] for j in targets])
    preds = model.predict(_augment(x_gb, exog))
    return pd.DataFrame({"datetime": dts[targets], "actual": values[targets, rtlo_idx], "forecast_mw": preds})


def predict_hindcast(
    recent_load: pd.DataFrame,
    recent_weather: pd.DataFrame | None = None,
    model: CatBoostRegressor | None = None,
    *,
    max_hours: int | None = None,
) -> pd.DataFrame:
    """1-hour-ahead rolling hindcast → ``[datetime, actual, forecast_mw]``."""
    if model is None:
        hm = load_horizon_models(1)
        model = hm[0] if hm else load_model()
    values, feature_cols, rtlo_idx, dts = _engineer(recent_load, recent_weather)
    if len(values) < LOOKBACK + 1:
        return pd.DataFrame(columns=_EMPTY)
    return _hindcast(values, feature_cols, rtlo_idx, dts, model, hi=0, max_hours=max_hours)


def predict_dayahead_hindcast(
    recent_load: pd.DataFrame,
    recent_weather: pd.DataFrame | None = None,
    *,
    max_hours: int | None = None,
) -> pd.DataFrame:
    """24-hour-ahead rolling hindcast (matches ISO's day-ahead horizon)."""
    hmodels = load_horizon_models()
    if hmodels is None:
        return pd.DataFrame(columns=_EMPTY)
    values, feature_cols, rtlo_idx, dts = _engineer(recent_load, recent_weather)
    if len(values) < LOOKBACK + HORIZON:
        return pd.DataFrame(columns=_EMPTY)
    return _hindcast(values, feature_cols, rtlo_idx, dts, hmodels[-1], hi=HORIZON - 1, max_hours=max_hours)


def predict_next_24h(
    recent_load: pd.DataFrame,
    recent_weather: pd.DataFrame,
    model: CatBoostRegressor | None = None,
    *,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Forward *horizon*-hour forecast → ``[datetime, forecast_mw]``.

    Uses the direct per-horizon models with target-hour exogenous features (forecast weather +
    calendar at t+h). Falls back to a recursive roll-out with the single model if per-horizon
    models are unavailable.
    """
    load = recent_load[["datetime", "RTLO"]].copy()
    load["datetime"] = pd.to_datetime(load["datetime"]).dt.floor("h")
    load = load.dropna(subset=["RTLO"]).drop_duplicates("datetime").sort_values("datetime")
    if len(load) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} hours of load, got {len(load)}")
    last_dt = load["datetime"].max()
    future_dates = pd.date_range(last_dt + pd.Timedelta(hours=1), periods=horizon, freq="h")

    hmodels = load_horizon_models(horizon)
    values, feature_cols, rtlo_idx, dts = _engineer(recent_load, recent_weather, future_dates=future_dates)
    exog_idx = target_exog_indices(feature_cols)

    obs = np.where(dts <= np.datetime64(last_dt))[0]
    if len(obs) < LOOKBACK:
        raise ValueError("Not enough engineered history for a forecast window")
    t = int(obs[-1])  # index of the last observed hour (the forecast origin)
    window = values[t - LOOKBACK + 1 : t + 1][np.newaxis, :, :]
    x_gb, _ = extract_lag_features(window, feature_cols, rtlo_idx)

    if hmodels is not None:
        preds: list[float] = []
        out_dates: list = []
        for hi, m in enumerate(hmodels):
            tr = t + 1 + hi
            if tr >= len(values):
                break
            x_aug = _augment(x_gb, values[tr, exog_idx][np.newaxis, :])
            preds.append(float(m.predict(x_aug)[0]))
            out_dates.append(dts[tr])
        return pd.DataFrame({"datetime": out_dates, "forecast_mw": preds})

    # ---- recursive fallback (single augmented model, h=1) ----
    if model is None:
        model = load_model()
    preds, out_dates = [], []
    for hi in range(horizon):
        tr = t + 1 + hi
        if tr >= len(values):
            break
        win = values[tr - LOOKBACK : tr][np.newaxis, :, :]
        xg, _ = extract_lag_features(win, feature_cols, rtlo_idx)
        x_aug = _augment(xg, values[tr, exog_idx][np.newaxis, :])
        p = float(model.predict(x_aug)[0])
        values[tr, rtlo_idx] = p  # feed back
        preds.append(p)
        out_dates.append(dts[tr])
    return pd.DataFrame({"datetime": out_dates, "forecast_mw": preds})
