"""Live comparison data for the dashboard — built from current ISO-NE feeds.

Replaces the stale, training-time ``backtest_results.parquet`` with a fresh frame computed
on demand from the ISO-NE Web Services API: actual real-time demand, the Beacon model's
rolling hindcast, and ISO-NE's day-ahead demand (the benchmark). Cached for one hour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from nema_forecast.config import LOOKBACK

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def build_recent_comparison(days: int = 30) -> pd.DataFrame:
    """Build a live ``[datetime, actual, catboost_pred, iso_forecast]`` comparison frame.

    * ``actual`` — ISO-NE real-time hourly demand (NEMA).
    * ``catboost_pred`` — Beacon's one-step-ahead hindcast for each hour.
    * ``iso_forecast`` — ISO-NE day-ahead hourly demand (the benchmark).

    The column name ``catboost_pred`` is kept for compatibility with the existing chart and
    metric helpers; it holds the Beacon model's predictions.
    """
    from nema_forecast.data.iso_ne_ws import (
        fetch_dayahead_demand_recent,
        fetch_realtime_demand_recent,
    )
    from nema_forecast.model.inference import load_model, predict_hindcast

    cols = ["datetime", "actual", "catboost_pred", "iso_forecast"]

    # Fetch enough actuals to cover the comparison window *plus* the lookback + publish lag.
    actual = fetch_realtime_demand_recent(days_back=days + 10)
    if actual.empty or len(actual) < LOOKBACK + 1:
        logger.warning("Not enough real-time demand to build comparison (%d rows)", len(actual))
        return pd.DataFrame(columns=cols)

    hind = predict_hindcast(actual, model=load_model(), max_hours=days * 24)
    if hind.empty:
        return pd.DataFrame(columns=cols)
    df = hind.rename(columns={"forecast_mw": "catboost_pred"})

    iso = fetch_dayahead_demand_recent(days_back=days + 5)
    if not iso.empty:
        df = df.merge(iso, on="datetime", how="left")
    else:
        df["iso_forecast"] = np.nan

    return df[cols].sort_values("datetime").reset_index(drop=True)
