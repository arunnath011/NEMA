"""Shared ledger-refresh logic: lock a fresh day-ahead forecast and score matured rows.

Used by both the manual script (``scripts/update_forecast_ledger.py``) and the deployed app
(``dashboard/live_data.maybe_update_and_load_ledger``). Pure of Streamlit and of any commit
mechanism — it only computes the next ledger state and whether it changed.
"""

from __future__ import annotations

import logging

import pandas as pd

from nema_forecast.config import LOOKBACK
from nema_forecast.data.iso_ne_ws import fetch_dayahead_demand_recent, fetch_realtime_demand_recent
from nema_forecast.data.open_meteo import fetch_recent_weather
from nema_forecast.forecast_ledger import append_forecasts, load_ledger, score_pending, to_csv
from nema_forecast.model.inference import load_model, predict_next_24h

logger = logging.getLogger(__name__)


def refresh_ledger() -> tuple[pd.DataFrame, bool]:
    """Return ``(ledger, changed)``: the ledger with a new locked forecast + newly-scored rows.

    ``changed`` is True only when the committed CSV bytes would differ — so callers commit at
    most once per real change. If ISO-NE demand is unavailable the existing ledger is returned
    unchanged (``changed=False``).
    """
    before = load_ledger()

    load = fetch_realtime_demand_recent(days_back=16)
    if load.empty or len(load) < LOOKBACK + 1:
        logger.warning("Ledger refresh: insufficient ISO-NE demand (%d rows); leaving ledger as-is.", len(load))
        return before, False

    weather = fetch_recent_weather(past_days=92, forecast_days=16)
    origin = pd.to_datetime(load["datetime"]).max()

    after = before
    fc = predict_next_24h(load, weather, model=load_model())
    if not fc.empty:
        target = pd.to_datetime(fc["datetime"])
        new_rows = pd.DataFrame(
            {
                "target_datetime": target,
                "forecast_mw": fc["forecast_mw"].to_numpy(),
                "forecast_made_at": pd.Timestamp.now(tz="UTC"),
                "origin_datetime": origin,
                "horizon_h": ((target - origin) / pd.Timedelta(hours=1)).round().astype(int),
            }
        )
        after = append_forecasts(before, new_rows)

    iso = fetch_dayahead_demand_recent(days_back=24)
    after = score_pending(after, load, iso)

    changed = to_csv(after) != to_csv(before)
    return after, changed
