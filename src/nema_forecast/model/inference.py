"""Inference pipeline — generate a real 24 h-ahead forecast from the trained model.

The model is trained to predict load one step ahead (h=1) from a 168 h lookback window.
To produce a full 24 h curve we forecast **recursively**: predict the next hour, write that
prediction back into the load history, and roll the window forward one hour at a time. Each
later step therefore consumes the model's own earlier predictions for any lag that now falls
inside the forecast horizon (the kept lags are ≥4 h, so the first few steps still rely purely
on observed load).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from nema_forecast.config import HORIZON, LOOKBACK, MODELS_DIR
from nema_forecast.data.preprocessing import apply_imputation, load_imputation_stats
from nema_forecast.features.engineering import engineer_features, extract_lag_features

logger = logging.getLogger(__name__)


def load_model(path: Path | None = None) -> CatBoostRegressor:
    path = path or MODELS_DIR / "catboost_model.cbm"
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def predict_next_24h(
    recent_load: pd.DataFrame,
    recent_weather: pd.DataFrame,
    model: CatBoostRegressor | None = None,
    *,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Produce a recursive *horizon*-hour-ahead forecast.

    Parameters
    ----------
    recent_load : DataFrame
        At least ``LOOKBACK`` rows of recent hourly load with columns ``[datetime, RTLO]``.
    recent_weather : DataFrame
        Hourly weather covering the recent window **and** the forecast horizon (e.g. the
        OpenWeatherMap 5-day forecast). Merged on ``datetime``.
    model : CatBoostRegressor, optional
        Pre-loaded model; loaded from disk if *None*.

    Returns
    -------
    DataFrame with columns ``[datetime, forecast_mw]`` (length *horizon*).
    """
    if model is None:
        model = load_model()
    stats = load_imputation_stats(MODELS_DIR / "imputation_stats.json")

    load = recent_load[["datetime", "RTLO"]].copy()
    load["datetime"] = pd.to_datetime(load["datetime"]).dt.floor("h")
    load = load.dropna(subset=["RTLO"]).drop_duplicates("datetime").sort_values("datetime")

    if len(load) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} hours of load, got {len(load)}")

    last_dt = load["datetime"].max()
    future_dates = pd.date_range(last_dt + pd.Timedelta(hours=1), periods=horizon, freq="h")

    # Future rows: RTLO unknown (placeholder = last observed; never read before it is
    # overwritten by a prediction — see recursion below — but keeps rows past dropna()).
    future = pd.DataFrame({"datetime": future_dates, "RTLO": load["RTLO"].iloc[-1]})
    full = pd.concat([load, future], ignore_index=True)

    # Attach weather (forecast covers the future hours) and impute any gaps.
    if recent_weather is not None and not recent_weather.empty:
        wx = recent_weather.copy()
        wx["datetime"] = pd.to_datetime(wx["datetime"]).dt.floor("h")
        wx = wx.drop_duplicates("datetime")
        full = full.merge(wx, on="datetime", how="left")
    full = apply_imputation(full, stats)

    feat = engineer_features(full)
    feature_cols = [c for c in feat.columns if c != "datetime"]
    rtlo_idx = feature_cols.index("RTLO")
    values = feat[feature_cols].values.astype(float)

    # Map each future timestamp to its row in the engineered frame.
    feat_dt = pd.to_datetime(feat["datetime"]).reset_index(drop=True)
    dt_to_row = {dt: i for i, dt in enumerate(feat_dt)}

    preds: list[float] = []
    out_dates: list[pd.Timestamp] = []
    for target_dt in future_dates:
        idx = dt_to_row.get(target_dt)
        if idx is None or idx < LOOKBACK:
            break  # not enough engineered history to form a window
        window = values[idx - LOOKBACK : idx][np.newaxis, :, :]
        x_gb, _ = extract_lag_features(window, feature_cols, rtlo_idx)
        pred = float(model.predict(x_gb)[0])
        values[idx, rtlo_idx] = pred  # feed back for subsequent steps
        preds.append(pred)
        out_dates.append(target_dt)

    return pd.DataFrame({"datetime": out_dates, "forecast_mw": preds})
