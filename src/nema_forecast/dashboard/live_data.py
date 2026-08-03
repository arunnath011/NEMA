"""Live comparison data for the dashboard — built from current ISO-NE feeds.

Replaces the stale, training-time ``backtest_results.parquet`` with a fresh frame computed
on demand from the ISO-NE Web Services API: actual real-time demand, the Beacon model's
rolling hindcast, and ISO-NE's day-ahead demand (the benchmark). Cached for one hour.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

from nema_forecast.config import HORIZON, LOOKBACK

logger = logging.getLogger(__name__)

_LEDGER_REPO_DEFAULT = "arunnath011/NEMA"


def _secret(name: str) -> str | None:
    """Read *name* from Streamlit secrets first, then the environment (None if unset/empty)."""
    try:
        val = st.secrets.get(name)  # type: ignore[no-untyped-call]
        if val:
            return str(val)
    except (FileNotFoundError, AttributeError):
        pass
    return os.getenv(name) or None


class LiveDataError(Exception):
    """A live feed the comparison depends on could not be loaded.

    The message names the specific source (ISO-NE demand vs Open-Meteo weather) so the page
    can show an accurate, actionable reason instead of a generic 'unavailable'.
    """


@st.cache_data(ttl=3600, show_spinner=False)
def get_model_weather() -> pd.DataFrame:
    """Open-Meteo recent + forecast hourly weather (the same source the model trains on).

    Raises on an empty result so a transient fetch failure is **not** cached — otherwise the
    day-ahead forecast would silently run weather-blind (much higher error) for a full hour.
    """
    from nema_forecast.data.open_meteo import fetch_recent_weather

    # forecast_days=8 (192 h) covers the day-ahead hindcast *and* the +4-day Outlook horizon.
    wx = fetch_recent_weather(past_days=92, forecast_days=8)
    if wx.empty:
        raise RuntimeError("Open-Meteo weather fetch returned no data")
    return wx


@st.cache_data(ttl=3600, show_spinner=False)
def build_recent_comparison(days: int = 30) -> pd.DataFrame:
    """Build a live ``[datetime, actual, catboost_pred, iso_forecast]`` comparison frame.

    * ``actual`` — ISO-NE real-time hourly demand (NEMA).
    * ``catboost_pred`` — Beacon's **day-ahead** (24 h) forecast, using Open-Meteo weather at
      the target hour — the horizon that matches ISO's published forecast (apples-to-apples).
    * ``iso_forecast`` — ISO-NE day-ahead hourly demand (the benchmark).

    The column name ``catboost_pred`` is kept for compatibility with the chart/metric helpers.
    """
    from nema_forecast.data.iso_ne_ws import (
        fetch_dayahead_demand_recent,
        fetch_realtime_demand_recent,
    )
    from nema_forecast.model.inference import predict_dayahead_hindcast

    cols = ["datetime", "actual", "catboost_pred", "iso_forecast"]

    # Fetch enough actuals to cover the comparison window *plus* the lookback + day-ahead offset.
    # ISO-NE demand needs credentials (or can be briefly rate-limited); surface that distinctly.
    actual = fetch_realtime_demand_recent(days_back=days + 12)
    if actual.empty or len(actual) < LOOKBACK + HORIZON + 1:
        logger.warning("Not enough real-time demand to build comparison (%d rows)", len(actual))
        raise LiveDataError(
            "the live **ISO-NE demand** feed returned no data — this needs the ISO-NE Web "
            "Services credentials (`ISO_NE_WS_USER` / `ISO_NE_WS_PASS`) set in Streamlit "
            "secrets, or it may be a brief rate limit."
        )

    # The day-ahead forecast is weather-dominated; without weather it is meaningless (MAE jumps
    # ~4x). Open-Meteo is keyless, so a failure here is a network/rate-limit issue, not a secret.
    try:
        weather = get_model_weather()
    except Exception as exc:
        logger.warning("Weather unavailable, skipping live comparison: %s", exc)
        raise LiveDataError(
            "the **Open-Meteo weather** feed could not be reached (it is keyless, so this is a "
            "temporary network/rate-limit issue on the host, not a credentials problem)."
        ) from exc

    hind = predict_dayahead_hindcast(actual, weather, max_hours=days * 24)
    if hind.empty:
        return pd.DataFrame(columns=cols)
    df = hind.rename(columns={"forecast_mw": "catboost_pred"})

    iso = fetch_dayahead_demand_recent(days_back=days + 5)
    if not iso.empty:
        df = df.merge(iso, on="datetime", how="left")
    else:
        df["iso_forecast"] = np.nan

    return df[cols].sort_values("datetime").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_outlook() -> pd.DataFrame:
    """Beacon's calendar-future Outlook: forecast for the hours after the latest ISO-NE actual.

    Returns ``[datetime, forecast_mw, lead_h]`` for leads up to the model's max (~+4 days). These
    hours' actuals are not published yet, so this is a genuine forward forecast. Empty if the
    Outlook model or the live feeds are unavailable.
    """
    from nema_forecast.data.iso_ne_ws import fetch_realtime_demand_recent
    from nema_forecast.model.inference import predict_outlook

    cols = ["datetime", "forecast_mw", "lead_h"]
    try:
        weather = get_model_weather()
    except Exception as exc:
        logger.warning("Outlook: weather unavailable (%s)", exc)
        return pd.DataFrame(columns=cols)
    actual = fetch_realtime_demand_recent(days_back=14)
    if actual.empty or len(actual) < LOOKBACK + 1:
        logger.warning("Outlook: insufficient ISO-NE demand (%d rows)", len(actual))
        return pd.DataFrame(columns=cols)
    return predict_outlook(actual, weather)


@st.cache_data(ttl=21600, show_spinner=False)
def maybe_update_and_load_ledger() -> pd.DataFrame:
    """Return the forecast ledger, refreshing + committing it first when a token is configured.

    ISO-NE blocks GitHub Actions, but this app can reach ISO-NE, so it maintains the ledger
    itself: it locks the newest day-ahead forecast, scores matured rows, and — when the ledger
    actually changes — commits the CSV via the GitHub API (``GH_LEDGER_TOKEN`` in secrets). The
    commit triggers a redeploy; on the next load there is no change, so it settles. Without a
    token it is read-only: it just returns the committed ledger. Cached (6 h) so the model/API
    work runs at most a few times a day.
    """
    from nema_forecast.forecast_ledger import load_ledger

    token = _secret("GH_LEDGER_TOKEN")
    if not token:
        return load_ledger()

    from nema_forecast.config import LEDGER_PATH, PROJECT_ROOT
    from nema_forecast.forecast_ledger import save_ledger, to_csv
    from nema_forecast.github_ledger import commit_text_file
    from nema_forecast.ledger_update import refresh_ledger

    try:
        ledger, changed = refresh_ledger()
    except Exception as exc:  # never let ledger maintenance break the page
        logger.warning("Ledger refresh failed: %s", exc)
        return load_ledger()

    if changed:
        repo = _secret("GH_LEDGER_REPO") or _LEDGER_REPO_DEFAULT
        path = LEDGER_PATH.relative_to(PROJECT_ROOT).as_posix()
        if commit_text_file(path, to_csv(ledger), "Update forecast ledger [skip ci]", token=token, repo=repo):
            save_ledger(ledger)  # reflect in this container until the redeploy picks up the commit
    return ledger
