"""Open-Meteo weather client — free historical + forecast hourly weather (no API key).

Open-Meteo provides:
  * the **archive** API (ERA5 reanalysis, back to 1940) — used for the training history and
    for scoring live day-ahead forecasts over past days, and
  * the **forecast** API (incl. ``past_days``) — used for recent + near-future hours when
    serving the live forecast.

Both are free and keyless. Crucially, this is the *same* weather source for training and
serving, which removes the train/serve weather-source mismatch (matching the source took
Beacon's live day-ahead MAE from ~147 down to ~90, on par with ISO-NE).

Fields are mapped to the columns the model expects: ``temp, humidity, wind_speed, dew_point,
clouds_all, feels_like, visibility`` (°F / mph / %). The archive API does not return
visibility, so it is filled with a constant (it was near-constant and low-signal in the
original data).
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from nema_forecast.config import BOSTON_LAT, BOSTON_LON, DATA_CACHE_DIR

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HOURLY = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,dew_point_2m," "cloud_cover,apparent_temperature,visibility"
)
_DEFAULT_VISIBILITY = 10000.0
_TZ = "America/New_York"

_COLUMNS = ["datetime", "temp", "humidity", "wind_speed", "dew_point", "clouds_all", "feels_like", "visibility"]


def _params(**extra: object) -> dict:
    return {
        "latitude": BOSTON_LAT,
        "longitude": BOSTON_LON,
        "hourly": _HOURLY,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": _TZ,
        **extra,
    }


def _parse(payload: dict) -> pd.DataFrame:
    h = payload.get("hourly")
    if not h or not h.get("time"):
        return pd.DataFrame(columns=_COLUMNS)
    vis = h.get("visibility") or [None] * len(h["time"])
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(h["time"]),
            "temp": h.get("temperature_2m"),
            "humidity": h.get("relative_humidity_2m"),
            "wind_speed": h.get("wind_speed_10m"),
            "dew_point": h.get("dew_point_2m"),
            "clouds_all": h.get("cloud_cover"),
            "feels_like": h.get("apparent_temperature"),
            "visibility": [v if v is not None else _DEFAULT_VISIBILITY for v in vis],
        }
    )
    df["visibility"] = df["visibility"].fillna(_DEFAULT_VISIBILITY)
    return df


def _get(url: str, params: dict, *, retries: int = 4) -> pd.DataFrame:
    """GET + parse an Open-Meteo response, retrying transient failures and rate limits.

    Weather is the dominant driver of the day-ahead forecast, so a failed fetch (which would
    silently fall back to median weather) is retried with exponential backoff, honouring any
    ``Retry-After`` header. Returns an empty frame only if every attempt fails.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            logger.warning("Open-Meteo connection error (%s); retry in %ds", exc, 2**attempt)
            time.sleep(2**attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = float(resp.headers.get("Retry-After", 2**attempt))
            logger.warning("Open-Meteo %d; backing off %.0fs (%d/%d)", resp.status_code, wait, attempt + 1, retries)
            time.sleep(min(wait, 30))
            continue
        try:
            resp.raise_for_status()
            return _parse(resp.json())
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Open-Meteo request failed (%s): %s", url, exc)
            return pd.DataFrame(columns=_COLUMNS)
    logger.warning("Open-Meteo gave up after %d retries: %s", retries, url)
    return pd.DataFrame(columns=_COLUMNS)


def fetch_archive_weather(start_date: str, end_date: str, *, force_refresh: bool = False) -> pd.DataFrame:
    """Historical hourly weather for [start_date, end_date] (YYYY-MM-DD). Cached to parquet."""
    cache = DATA_CACHE_DIR / f"openmeteo_archive_{start_date}_{end_date}.parquet"
    if cache.exists() and not force_refresh:
        return pd.read_parquet(cache)
    df = _get(ARCHIVE_URL, _params(start_date=start_date, end_date=end_date))
    if not df.empty:
        df.to_parquet(cache, index=False)
    return df


def fetch_recent_weather(past_days: int = 92, forecast_days: int = 5) -> pd.DataFrame:
    """Recent + near-future hourly weather (forecast API with ``past_days``).

    Covers the live hindcast window (past) and the forward forecast horizon (future) in one
    call. Returns ``[datetime, temp, humidity, wind_speed, dew_point, clouds_all, feels_like,
    visibility]``.
    """
    return _get(FORECAST_URL, _params(past_days=min(past_days, 92), forecast_days=min(forecast_days, 16)))
