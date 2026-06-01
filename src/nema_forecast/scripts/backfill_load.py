"""One-time historical backfill of NEMA demand from the ISO-NE Web Services API.

Pulls real-time hourly demand (and optionally day-ahead demand) one day at a time and
writes a consolidated Parquet that ``model.train`` / ``model.backtest`` consume. This is
the source of the training series, so train and serve use the *identical* signal.

Usage::

    # validate the pipeline on a short range first, then widen
    python -m nema_forecast.scripts.backfill_load --start 2023-01-01 --end 2025-05-31
    python -m nema_forecast.scripts.backfill_load --start 2017-03-01 --with-dayahead

Requires ISO_NE_WS_USER / ISO_NE_WS_PASS in your environment (.env). The run is resumable:
per-day responses are cached, so re-running skips days already fetched. ~3000 calls cover
2017→today; a small delay between calls keeps it rate-friendly.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from nema_forecast.config import DATA_CACHE_DIR
from nema_forecast.data.iso_ne_ws import fetch_dayahead_demand_day, fetch_realtime_demand_day

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

LOAD_HISTORY_PATH = DATA_CACHE_DIR / "load_history.parquet"
ISO_FORECAST_HISTORY_PATH = DATA_CACHE_DIR / "iso_forecast_history.parquet"


def _daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def backfill(
    start: datetime,
    end: datetime,
    *,
    with_dayahead: bool = False,
    delay: float = 0.15,
) -> None:
    """Loop days in [start, end], fetch demand, and write consolidated Parquets."""
    rt_frames: list[pd.DataFrame] = []
    da_frames: list[pd.DataFrame] = []

    total = (end - start).days + 1
    for i, day in enumerate(_daterange(start, end), 1):
        try:
            rt = fetch_realtime_demand_day(day)
            if not rt.empty:
                rt_frames.append(rt)
            if with_dayahead:
                da = fetch_dayahead_demand_day(day)
                if not da.empty:
                    da_frames.append(da)
        except Exception as exc:
            logger.warning("Skipping %s: %s", day.date(), exc)

        if i % 30 == 0 or i == total:
            logger.info("Progress: %d/%d days (%s)", i, total, day.date())
        time.sleep(delay)

    _write(rt_frames, LOAD_HISTORY_PATH, "RTLO", "real-time demand")
    if with_dayahead:
        _write(da_frames, ISO_FORECAST_HISTORY_PATH, "iso_forecast", "day-ahead demand")


def _write(frames: list[pd.DataFrame], path, value_col: str, label: str) -> None:
    if not frames:
        logger.warning("No %s data fetched — nothing written to %s", label, path)
        return
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
    df = df.reset_index(drop=True)
    df.to_parquet(path, index=False)
    logger.info(
        "Wrote %d rows of %s → %s (%s → %s)",
        len(df),
        label,
        path,
        df["datetime"].min(),
        df["datetime"].max(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NEMA demand from ISO-NE Web Services")
    parser.add_argument("--start", default="2017-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD); defaults to today")
    parser.add_argument(
        "--with-dayahead",
        action="store_true",
        help="Also backfill day-ahead demand (the ISO-NE benchmark)",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="Seconds between API calls")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()

    logger.info("Backfilling %s → %s (day-ahead=%s)", start.date(), end.date(), args.with_dayahead)
    backfill(start, end, with_dayahead=args.with_dayahead, delay=args.delay)


if __name__ == "__main__":
    main()
