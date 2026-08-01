"""Refresh the committed recent-weather snapshot from Open-Meteo.

Run locally or from CI (GitHub Actions) — anywhere with an IP Open-Meteo does not rate-limit.
Writes ``data/weather_snapshot.parquet``, which the dashboard falls back to when a live fetch
fails on the deploy host (Streamlit Cloud's shared IP is rate-limited by Open-Meteo).

    python scripts/refresh_weather_snapshot.py
"""

from __future__ import annotations

import sys

from nema_forecast.config import WEATHER_SNAPSHOT_PATH
from nema_forecast.data.open_meteo import fetch_recent_weather, save_weather_snapshot


def main() -> int:
    # Fetch directly from the live API (no snapshot fallback), with patient retries since CI is
    # not latency-sensitive. 92 past days + 16 forecast days maximises the window the snapshot
    # can serve before the sliding comparison window outruns it.
    df = fetch_recent_weather(past_days=92, forecast_days=16, retries=5, allow_snapshot=False)
    if df.empty:
        print("ERROR: Open-Meteo returned no data; snapshot not updated.", file=sys.stderr)
        return 1
    save_weather_snapshot(df)
    print(
        f"Wrote {len(df):,} hourly rows to {WEATHER_SNAPSHOT_PATH} "
        f"({df['datetime'].min()} → {df['datetime'].max()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
