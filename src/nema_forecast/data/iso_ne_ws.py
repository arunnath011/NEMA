"""ISO New England Web Services API client — near-real-time NEMA demand.

This is the *live* data source for the dashboard. It replaces the old WHLSECOST CSV
report (a settlement report published on a 4-6 week delay) with the ISO-NE Web Services
REST API, which exposes near-real-time (≈1 hour latency) and day-ahead hourly demand by
load zone.

Endpoints used (base ``https://webservices.iso-ne.com/api/v1.1``, HTTP Basic auth over SSL):

* ``/realtimehourlydemand/day/{YYYYMMDD}/location/{locId}`` — metered real-time demand
* ``/dayaheadhourlydemand/day/{YYYYMMDD}/location/{locId}`` — ISO's day-ahead demand
  (used as the benchmark the CatBoost model is compared against)

The real-time demand series is exposed under the column name ``RTLO`` so the rest of the
pipeline (features, preprocessing, inference) needs no renaming — note that it now holds
real-time metered demand in MW, not the legacy wholesale-cost RTLO value.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from nema_forecast.config import (
    DATA_CACHE_DIR,
    ISO_NE_LOCATION_ID,
    ISO_NE_WS_BASE_URL,
    ISO_NE_WS_PASS,
    ISO_NE_WS_USER,
)

logger = logging.getLogger(__name__)

_SESSION: requests.Session | None = None

# Timezone of ISO-NE BeginDate timestamps. We convert to local wall-clock and drop the
# offset so the series aligns with the calendar features (which key off local hour).
_ISO_NE_TZ = "America/New_York"


def _ws_session() -> requests.Session:
    """Return a cached requests session authenticated for the ISO-NE Web Services API."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    if not ISO_NE_WS_USER or not ISO_NE_WS_PASS:
        raise RuntimeError(
            "ISO-NE Web Services credentials missing. Set ISO_NE_WS_USER and "
            "ISO_NE_WS_PASS in your .env (local) or Streamlit secrets (cloud)."
        )

    sess = requests.Session()
    sess.auth = HTTPBasicAuth(ISO_NE_WS_USER, ISO_NE_WS_PASS)
    sess.headers.update({"Accept": "application/json"})
    _SESSION = sess
    return _SESSION


# ---------------------------------------------------------------------------
# Real-time hourly demand (the live load series → column "RTLO")
# ---------------------------------------------------------------------------


def fetch_realtime_demand_day(
    date: datetime,
    location_id: int = ISO_NE_LOCATION_ID,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch one day of real-time hourly demand for a load zone.

    Returns a DataFrame with columns ``[datetime, RTLO]`` (empty if the day has no data
    yet, e.g. a future date or one not published). Cached per-day under ``data/cache``.
    """
    day_str = date.strftime("%Y%m%d")
    cache_path = DATA_CACHE_DIR / f"rtdemand_{location_id}_{day_str}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    url = f"{ISO_NE_WS_BASE_URL}/realtimehourlydemand/day/{day_str}/location/{location_id}.json"
    df = _fetch_and_parse(url, wrapper="HourlyRtDemands", item="HourlyRtDemand", value_col="RTLO")

    # Cache non-empty days only (don't pin an empty result for a day that may publish later).
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_realtime_demand_recent(
    days_back: int = 10,
    location_id: int = ISO_NE_LOCATION_ID,
) -> pd.DataFrame:
    """Fetch the most recent *days_back* days of real-time demand, concatenated.

    The current and previous day are always force-refreshed (they fill in through the
    day); older days come from cache. Returns ``[datetime, RTLO]`` sorted ascending.
    """
    today = datetime.now().date()
    frames: list[pd.DataFrame] = []
    for offset in range(days_back, -1, -1):
        d = today - timedelta(days=offset)
        force = offset <= 1
        try:
            df = fetch_realtime_demand_day(datetime(d.year, d.month, d.day), location_id, force_refresh=force)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("Real-time demand fetch failed for %s: %s", d, exc)

    return _combine(frames)


# ---------------------------------------------------------------------------
# Day-ahead hourly demand (the ISO-NE benchmark → column "iso_forecast")
# ---------------------------------------------------------------------------


def fetch_dayahead_demand_day(
    date: datetime,
    location_id: int = ISO_NE_LOCATION_ID,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch one day of day-ahead hourly demand. Returns ``[datetime, iso_forecast]``."""
    day_str = date.strftime("%Y%m%d")
    cache_path = DATA_CACHE_DIR / f"dademand_{location_id}_{day_str}.parquet"

    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    url = f"{ISO_NE_WS_BASE_URL}/dayaheadhourlydemand/day/{day_str}/location/{location_id}.json"
    df = _fetch_and_parse(url, wrapper="HourlyDaDemands", item="HourlyDaDemand", value_col="iso_forecast")
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def fetch_dayahead_demand_recent(
    days_back: int = 60,
    location_id: int = ISO_NE_LOCATION_ID,
) -> pd.DataFrame:
    """Fetch the most recent *days_back* days of day-ahead demand, concatenated."""
    today = datetime.now().date()
    frames: list[pd.DataFrame] = []
    for offset in range(days_back, -1, -1):
        d = today - timedelta(days=offset)
        try:
            df = fetch_dayahead_demand_day(datetime(d.year, d.month, d.day), location_id)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.warning("Day-ahead demand fetch failed for %s: %s", d, exc)
        time.sleep(0.1)

    return _combine(frames, value_col="iso_forecast")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_and_parse(url: str, *, wrapper: str, item: str, value_col: str) -> pd.DataFrame:
    """GET *url* and parse an ISO-NE hourly-demand JSON payload."""
    try:
        resp = _ws_session().get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("ISO-NE WS request failed (%s): %s", url, exc)
        return pd.DataFrame(columns=["datetime", value_col])
    except ValueError as exc:  # JSON decode
        logger.warning("ISO-NE WS returned non-JSON (%s): %s", url, exc)
        return pd.DataFrame(columns=["datetime", value_col])

    return parse_demand(payload, wrapper=wrapper, item=item, value_col=value_col)


def parse_demand(payload: dict, *, wrapper: str, item: str, value_col: str) -> pd.DataFrame:
    """Parse an ISO-NE hourly-demand JSON payload into ``[datetime, value_col]``.

    Defensive against the ISO-NE quirk where a wrapper with a single hour returns a bare
    object instead of a list, and against missing keys / empty days. Pure function — no
    network — so it is unit-testable offline.
    """
    empty = pd.DataFrame(columns=["datetime", value_col])
    if not isinstance(payload, dict):
        return empty

    container = payload.get(wrapper) or {}
    rows = container.get(item) if isinstance(container, dict) else None
    if rows is None:
        return empty
    if isinstance(rows, dict):  # single-hour payload
        rows = [rows]
    if not rows:
        return empty

    records = []
    for r in rows:
        if not isinstance(r, dict) or "BeginDate" not in r or "Load" not in r:
            continue
        records.append({"datetime": r["BeginDate"], value_col: r["Load"]})
    if not records:
        return empty

    df = pd.DataFrame.from_records(records)
    # BeginDate is ISO-8601 with an Eastern offset; normalise to local wall-clock, naive.
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = dt.dt.tz_convert(_ISO_NE_TZ).dt.tz_localize(None)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["datetime", value_col])
    return df.sort_values("datetime").reset_index(drop=True)


def _combine(frames: list[pd.DataFrame], value_col: str = "RTLO") -> pd.DataFrame:
    """Concatenate per-day frames, dedupe on datetime, sort ascending."""
    if not frames:
        return pd.DataFrame(columns=["datetime", value_col])
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.drop_duplicates(subset=["datetime"], keep="last")
    return combined.sort_values("datetime").reset_index(drop=True)
