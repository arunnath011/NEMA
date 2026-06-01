"""OpenWeatherMap data fetching — current, forecast, and historical weather for Boston."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from nema_forecast.config import (
    BOSTON_LAT,
    BOSTON_LON,
    DATA_CACHE_DIR,
    LEGACY_DATA_PATH,
    OWM_API_KEY,
    OWM_CURRENT_URL,
    OWM_FORECAST_URL,
)

logger = logging.getLogger(__name__)

_SESSION = requests.Session()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_current_weather() -> dict:
    """Get current Boston weather from OpenWeatherMap (imperial units → °F)."""
    params: dict[str, Any] = {
        "lat": BOSTON_LAT,
        "lon": BOSTON_LON,
        "appid": OWM_API_KEY,
        "units": "imperial",
    }
    resp = _SESSION.get(OWM_CURRENT_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return _flatten_owm_current(data)


def fetch_weather_forecast() -> pd.DataFrame:
    """Get 5-day / 3-hour Boston weather forecast (imperial → °F).

    Returns hourly-interpolated DataFrame with the same columns as historical weather.
    """
    params: dict[str, Any] = {
        "lat": BOSTON_LAT,
        "lon": BOSTON_LON,
        "appid": OWM_API_KEY,
        "units": "imperial",
    }
    resp = _SESSION.get(OWM_FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data.get("list", []):
        rows.append(
            {
                "datetime": pd.to_datetime(item["dt"], unit="s"),
                "temp": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "humidity": item["main"]["humidity"],
                "pressure": item["main"]["pressure"],
                "wind_speed": item["wind"]["speed"],
                "wind_deg": item["wind"].get("deg", 0),
                "clouds_all": item["clouds"]["all"],
                "visibility": item.get("visibility", 10000),
                "weather_main": item["weather"][0]["main"] if item.get("weather") else "",
                "weather_description": item["weather"][0]["description"] if item.get("weather") else "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Resample to hourly and forward-fill
    df = df.set_index("datetime").resample("1h").first().ffill().bfill().reset_index()
    return df


def load_historical_weather(
    source: str = "auto",
    extend_to: datetime | None = None,
) -> pd.DataFrame:
    """Load historical Boston weather.

    *source* can be ``"legacy"`` (read from NEMA_DATA_PATH csvs), ``"cache"``
    (previously fetched), or ``"auto"`` (try cache first, then legacy).

    If *extend_to* is provided and the weather data ends before that date,
    forward-fills the last observation to cover the gap (the OWM history API
    requires a paid plan, so this is the best free-tier approach).
    """
    cache_path = DATA_CACHE_DIR / "weather_history.parquet"

    if source in ("auto", "cache") and cache_path.exists():
        logger.info("Loading cached weather history")
        df = pd.read_parquet(cache_path)
    elif source in ("auto", "legacy"):
        df = _load_legacy_weather()
        if not df.empty:
            df.to_parquet(cache_path, index=False)
    else:
        df = pd.DataFrame()

    if df.empty:
        return df

    # Extend weather data to cover recent months where we lack observations
    if extend_to is None:
        extend_to = datetime.now()

    last_weather = pd.to_datetime(df["datetime"].max())
    if last_weather < pd.to_datetime(extend_to):
        gap_hours = pd.date_range(last_weather + pd.Timedelta(hours=1), extend_to, freq="h")
        if len(gap_hours) > 0:
            logger.info(
                "Extending weather data with forward-fill: %s → %s (%d hours)",
                last_weather,
                extend_to,
                len(gap_hours),
            )
            last_row = df.iloc[-1:].copy()
            extension = pd.DataFrame({"datetime": gap_hours})
            for col in df.columns:
                if col != "datetime":
                    extension[col] = last_row[col].values[0]
            df = pd.concat([df, extension], ignore_index=True)

    return df


def save_weather_snapshot(data: dict) -> None:
    """Append a current-weather snapshot to the daily cache file."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    path = DATA_CACHE_DIR / f"weather_snapshots_{today}.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps(data) + "\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_owm_current(raw: dict) -> dict:
    """Flatten the nested OWM current-weather JSON into a row dict."""
    main = raw.get("main", {})
    wind = raw.get("wind", {})
    clouds = raw.get("clouds", {})
    weather = raw["weather"][0] if raw.get("weather") else {}
    return {
        "datetime": datetime.utcfromtimestamp(raw["dt"]) + timedelta(seconds=abs(raw.get("timezone", -18000))),
        "temp": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "visibility": raw.get("visibility", 10000),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "clouds_all": clouds.get("all"),
        "dew_point": None,
        "weather_main": weather.get("main", ""),
        "weather_description": weather.get("description", ""),
    }


def _load_legacy_weather() -> pd.DataFrame:
    """Load weather CSVs from the original NEMA 3 project folder."""
    frames: list[pd.DataFrame] = []
    for name in ("weather_train.csv", "weather_test.csv"):
        p = LEGACY_DATA_PATH / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "local_datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["local_datetime"])
        elif "dt_iso" in df.columns:
            df["datetime"] = pd.to_datetime(df["dt_iso"].str.replace(r" \+\d{4} UTC", "", regex=True))
            if "timezone" in df.columns:
                df["datetime"] = df["datetime"] + pd.to_timedelta(df["timezone"].abs(), unit="s")
        frames.append(df)

    if not frames:
        logger.warning("No legacy weather files found at %s", LEGACY_DATA_PATH)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    keep_cols = [
        "datetime",
        "temp",
        "feels_like",
        "humidity",
        "pressure",
        "visibility",
        "dew_point",
        "wind_speed",
        "wind_deg",
        "clouds_all",
        "weather_main",
        "weather_description",
    ]
    keep_cols = [c for c in keep_cols if c in combined.columns]
    return combined[keep_cols]
