"""ISO New England data fetching — hourly wholesale load cost and three-day forecast."""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from nema_forecast.config import (
    DATA_CACHE_DIR,
    ISO_NE_BASE_URL,
    ISO_NE_FORECAST_BASE_URL,
    ISO_NE_FORECAST_REGION,
    ISO_NE_FORECAST_REPORT_PAGE,
    ISO_NE_LOCATION_ID,
    LEGACY_DATA_PATH,
)

logger = logging.getLogger(__name__)

_SESSION: requests.Session | None = None

_ISO_NE_REPORT_PAGE = (
    "https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/" "-/tree/whlsecost-hourly-nemassbost"
)


def _get_session() -> requests.Session:
    """Return a requests session with valid ISO-NE cookies.

    ISO-NE returns 403 on CSV downloads unless you first visit the report
    listing page to establish a session cookie.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    _SESSION = requests.Session()
    _SESSION.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/html,application/xhtml+xml,*/*",
        }
    )
    try:
        logger.info("Establishing ISO-NE session …")
        resp = _SESSION.get(_ISO_NE_REPORT_PAGE, timeout=30)
        resp.raise_for_status()
        logger.info("ISO-NE session established (cookies: %d)", len(_SESSION.cookies))
    except requests.RequestException as exc:
        logger.warning("Could not establish ISO-NE session: %s", exc)

    return _SESSION


# ---------------------------------------------------------------------------
# Hourly wholesale load cost (RTLO)
# ---------------------------------------------------------------------------


def fetch_iso_month(year: int, month: int, *, force_refresh: bool = False) -> pd.DataFrame:
    """Download a single month of NEMA hourly wholesale load data from ISO-NE.

    Falls back to local ``NEMA_DATA_PATH`` CSV files when the download fails.
    If a cached file exists but contains no data rows, also falls back.
    """
    yyyymm = f"{year}{month:02d}"
    cache_path = DATA_CACHE_DIR / f"whlsecost_hourly_{ISO_NE_LOCATION_ID}_{yyyymm}.csv"

    if cache_path.exists() and not force_refresh:
        df_cached = _parse_whlsecost_csv(cache_path)
        if not df_cached.empty:
            logger.info("Cache hit: %s (%d rows)", cache_path.name, len(df_cached))
            return df_cached
        logger.info("Cache empty for %s, trying local fallback", cache_path.name)
        local = _load_local_whlsecost(yyyymm)
        if not local.empty:
            return local

    url = f"{ISO_NE_BASE_URL}?month={yyyymm}&locationId={ISO_NE_LOCATION_ID}"
    logger.info("Fetching ISO-NE data: %s", url)

    try:
        session = _get_session()
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        if resp.text.strip().startswith('"C"'):
            df_downloaded = _parse_whlsecost_csv_text(resp.text)
            if not df_downloaded.empty:
                cache_path.write_text(resp.text, encoding="utf-8")
                logger.info("Downloaded and cached: %s (%d rows)", cache_path.name, len(df_downloaded))
                return df_downloaded
            logger.warning("ISO-NE returned CSV with 0 data rows for %s (report not yet published?)", yyyymm)
            return pd.DataFrame()
        logger.warning("ISO-NE returned non-CSV for %s", yyyymm)
    except requests.RequestException:
        pass

    logger.warning("ISO-NE download failed for %s, trying local fallback", yyyymm)
    return _load_local_whlsecost(yyyymm)


def fetch_iso_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch multiple months of ISO-NE data and concatenate."""
    frames: list[pd.DataFrame] = []
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)

    while current <= end:
        try:
            df = fetch_iso_month(current.year, current.month, force_refresh=force_refresh)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.warning("Skipping %s-%02d", current.year, current.month)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if not frames:
        raise RuntimeError("No ISO-NE data could be loaded")
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def fetch_recent_load(months_back: int = 3) -> pd.DataFrame:
    """Fetch the most recent *months_back* months of load data.

    Force-refreshes the current month and two prior months (WHLSECOST
    reports are delayed ~4-6 weeks, so recent months may gain data between
    visits).  Uses cache for older months.
    """
    now = datetime.now()
    frames: list[pd.DataFrame] = []

    for offset in range(months_back, -1, -1):
        y = now.year
        m = now.month - offset
        while m <= 0:
            m += 12
            y -= 1
        force = offset <= 2
        try:
            df = fetch_iso_month(y, m, force_refresh=force)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.warning("Could not fetch %d-%02d", y, m)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Three-day reliability region demand forecast
# ---------------------------------------------------------------------------


def fetch_iso_forecast_day(date: datetime) -> pd.DataFrame:
    """Download the ISO-NE three-day forecast for a single target date.

    Returns a DataFrame with columns: datetime, mw, published_date
    filtered to the NEMASSBOST region, keeping only the most recent
    published forecast per hour.
    """
    date_str = date.strftime("%Y%m%d")
    cache_path = DATA_CACHE_DIR / f"iso_forecast_{date_str}.csv"

    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
    else:
        url = f"{ISO_NE_FORECAST_BASE_URL}?start={date_str}"
        logger.info("Fetching ISO forecast: %s", url)
        try:
            session = _get_session()
            # The forecast page needs its own cookie context
            session.get(ISO_NE_FORECAST_REPORT_PAGE, timeout=30)
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text
            if text.strip().startswith('"C"'):
                cache_path.write_text(text, encoding="utf-8")
            else:
                logger.warning("Non-CSV response for forecast %s", date_str)
                return pd.DataFrame()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch ISO forecast for %s: %s", date_str, exc)
            return pd.DataFrame()

    return _parse_forecast_csv_text(text)


def fetch_recent_iso_forecasts(days_back: int = 60) -> pd.DataFrame:
    """Fetch ISO-NE three-day forecasts for the last *days_back* days.

    Combines local legacy files with live downloads, de-duplicates, and
    returns a single DataFrame.
    """
    # Start with legacy local files
    legacy = load_iso_forecasts_local()

    # Fetch recent days from the API
    frames: list[pd.DataFrame] = []
    today = datetime.now().date()
    for i in range(days_back):
        d = today - timedelta(days=i)
        try:
            df = fetch_iso_forecast_day(datetime(d.year, d.month, d.day))
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if frames:
        live = pd.concat(frames, ignore_index=True)
    else:
        live = pd.DataFrame()

    # Combine legacy + live
    parts = [p for p in [legacy, live] if not p.empty]
    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.sort_values("published_date", ascending=False).drop_duplicates("datetime", keep="first")
    return combined.sort_values("datetime").reset_index(drop=True)


def load_iso_forecasts_local(folder: Path | None = None) -> pd.DataFrame:
    """Load all ISO three-day forecast CSVs from *folder* (or legacy path)."""
    if folder is None:
        folder = LEGACY_DATA_PATH / "IsoThreeDayForecast"
    if not folder.exists():
        logger.warning("ISO forecast folder not found: %s", folder)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for fpath in sorted(folder.glob("threedayreliabilityregionloadforecastreport_*.csv")):
        try:
            df = pd.read_csv(
                fpath,
                skiprows=5,
                header=None,
                names=["type", "forecast_date", "hour", "region", "mw", "pct", "published_date"],
            )
            df = df[(df["type"] == "D") & (df["region"] == ISO_NE_FORECAST_REGION)]
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    iso = pd.concat(frames, ignore_index=True)
    return _normalise_forecast_df(iso)


def load_iso_forecasts(folder: Path | None = None) -> pd.DataFrame:
    """Load ISO forecasts — combines local files + live API downloads."""
    return fetch_recent_iso_forecasts(days_back=60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_forecast_csv_text(text: str) -> pd.DataFrame:
    """Parse ISO-NE forecast CSV text (in-memory) into a clean DataFrame."""
    try:
        raw = pd.read_csv(
            io.StringIO(text),
            skiprows=5,
            header=None,
            names=[
                "type",
                "forecast_date",
                "hour",
                "region",
                "mw",
                "pct",
                "published_date",
            ],
        )
    except Exception:
        return pd.DataFrame()

    raw = raw[(raw["type"] == "D") & (raw["region"] == ISO_NE_FORECAST_REGION)]
    if raw.empty:
        return pd.DataFrame()

    return _normalise_forecast_df(raw)


def _normalise_forecast_df(iso: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing for forecast DataFrames."""
    iso = iso.copy()
    iso["forecast_date"] = pd.to_datetime(iso["forecast_date"], format="%m/%d/%Y")
    iso["published_date"] = pd.to_datetime(iso["published_date"], format="%m/%d/%Y %H:%M:%S")
    iso["hour"] = iso["hour"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    iso = iso[iso["hour"] != ""]
    iso["hour"] = iso["hour"].astype(int)
    iso["mw"] = pd.to_numeric(iso["mw"], errors="coerce")
    iso["datetime"] = iso["forecast_date"] + pd.to_timedelta(iso["hour"] - 1, unit="h")

    iso = iso.sort_values("published_date", ascending=False).drop_duplicates("datetime", keep="first")
    return iso.sort_values("datetime").reset_index(drop=True)


def _parse_whlsecost_csv_text(text: str) -> pd.DataFrame:
    """Parse ISO-NE wholesale cost CSV from an in-memory string."""
    try:
        raw = pd.read_csv(io.StringIO(text), skiprows=4, header=0)
    except Exception:
        return pd.DataFrame()

    if raw.empty or (raw.iloc[:, 0].dtype == object and (raw.iloc[:, 0] == "D").sum() == 0):
        return pd.DataFrame()

    raw = raw[raw.iloc[:, 0] == "D"].copy()
    raw.columns = raw.columns.str.strip().str.replace(" ", "_")

    rename_map = {
        "H": "RowType",
        "Location_ID": "LocationID",
        "Local_Date": "LocalDate",
        "Local_Hour": "LocalHour",
    }
    raw = raw.rename(columns=rename_map)

    raw["LocalHour"] = pd.to_numeric(raw["LocalHour"], errors="coerce")
    raw = raw.dropna(subset=["LocalHour"])
    raw["LocalHour"] = raw["LocalHour"].astype(int) - 1  # 1-24 → 0-23

    raw["datetime"] = pd.to_datetime(
        raw["LocalDate"] + " " + raw["LocalHour"].astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    raw["RTLO"] = pd.to_numeric(raw["RTLO"], errors="coerce")
    return raw.sort_values("datetime").reset_index(drop=True)


def _parse_whlsecost_csv(path: Path) -> pd.DataFrame:
    """Parse the ISO-NE wholesale cost CSV format (4 comment rows, then H/D rows)."""
    try:
        raw = pd.read_csv(path, skiprows=4, header=0)
    except Exception:
        return pd.DataFrame()

    if raw.empty or (raw.iloc[:, 0].dtype == object and (raw.iloc[:, 0] == "D").sum() == 0):
        return pd.DataFrame()

    raw = raw[raw.iloc[:, 0] == "D"].copy()
    raw.columns = raw.columns.str.strip().str.replace(" ", "_")

    rename_map = {
        "H": "RowType",
        "Location_ID": "LocationID",
        "Local_Date": "LocalDate",
        "Local_Hour": "LocalHour",
    }
    raw = raw.rename(columns=rename_map)

    raw["LocalHour"] = pd.to_numeric(raw["LocalHour"], errors="coerce")
    raw = raw.dropna(subset=["LocalHour"])
    raw["LocalHour"] = raw["LocalHour"].astype(int) - 1  # 1-24 → 0-23

    raw["datetime"] = pd.to_datetime(
        raw["LocalDate"] + " " + raw["LocalHour"].astype(str).str.zfill(2) + ":00:00",
        format="%m/%d/%Y %H:%M:%S",
    )
    raw["RTLO"] = pd.to_numeric(raw["RTLO"], errors="coerce")
    return raw.sort_values("datetime").reset_index(drop=True)


def _load_local_whlsecost(yyyymm: str) -> pd.DataFrame:
    """Fallback: load from the original NEMA 3 project folder."""
    for folder in ("train", "test"):
        p = LEGACY_DATA_PATH / folder / f"whlsecost_hourly_{ISO_NE_LOCATION_ID}_{yyyymm}.csv"
        if p.exists():
            logger.info("Loaded local fallback: %s", p)
            return _parse_whlsecost_csv(p)

    logger.warning("No data found for %s", yyyymm)
    return pd.DataFrame()
