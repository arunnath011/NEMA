"""Centralised configuration — paths, API endpoints, model constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st

        return str(st.secrets.get(key, os.getenv(key, default)))
    except (ImportError, AttributeError, FileNotFoundError):
        return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_DATA_PATH = Path(os.getenv("NEMA_DATA_PATH", ""))

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
OWM_API_KEY: str = _get_secret("OWM_API_KEY")

# ISO-NE Web Services API credentials (register at https://www.iso-ne.com/participate/
# support/web-services-data). Basic-auth username/password.
ISO_NE_WS_USER: str = _get_secret("ISO_NE_WS_USER")
ISO_NE_WS_PASS: str = _get_secret("ISO_NE_WS_PASS")

# EIA Open Data API key — the no-wait fallback source for live NEMA demand. A free key is
# issued instantly at https://www.eia.gov/opendata/register.php (unlike ISO-NE Web Services,
# which needs account approval). Provides true NEMA-zone hourly demand at ~1-2 h latency.
EIA_API_KEY: str = _get_secret("EIA_API_KEY")

# ---------------------------------------------------------------------------
# ISO-NE Web Services API (the live, near-real-time source)
# ---------------------------------------------------------------------------
ISO_NE_WS_BASE_URL = "https://webservices.iso-ne.com/api/v1.1"
ISO_NE_LOCATION_ID = 4008  # NEMA / Northeast Massachusetts and Boston (NEMASSBOST)

# ---------------------------------------------------------------------------
# EIA Open Data API (fallback live source — true NEMA-zone hourly demand)
# ---------------------------------------------------------------------------
EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/"
EIA_ISNE_PARENT = "ISNE"  # ISO New England balancing authority
EIA_NEMA_SUBBA = "4008"  # NEMASSBOST subregion (legacy series EBA.ISNE-4008.D.H)

# ---------------------------------------------------------------------------
# Legacy ISO-NE CSV report endpoints — DEPRECATED.
# The WHLSECOST report is a settlement report published on a 4-6 week delay and the
# three-day-forecast scrape needs HTML-cookie priming. Both are superseded by the Web
# Services API above and retained only for reference / one-off historical CSV imports.
# ---------------------------------------------------------------------------
ISO_NE_BASE_URL = "https://www.iso-ne.com/transform/csv/whlsecost/hourly"

ISO_NE_FORECAST_BASE_URL = "https://www.iso-ne.com/transform/csv/reliabilityregionloadforecast"

ISO_NE_FORECAST_REPORT_PAGE = (
    "https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/"
    "-/tree/three-day-reliability-region-demand-forecast"
)
ISO_NE_FORECAST_REGION = ".Z.NEMASSBOST"

# ---------------------------------------------------------------------------
# OpenWeatherMap endpoints
# ---------------------------------------------------------------------------
OWM_HISTORY_URL = "https://history.openweathermap.org/data/2.5/history/city"
OWM_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
BOSTON_LAT = 42.3656
BOSTON_LON = -71.0096

# ---------------------------------------------------------------------------
# Model hyper-parameters (mirroring the notebook)
# ---------------------------------------------------------------------------
LOOKBACK = 168  # hours (one week)
HORIZON = 24  # hours ahead
TRAIN_CUTOFF = "2024-12-31 23:00:00"

CATBOOST_PARAMS: dict = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "MAE",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "verbose": 0,
}

LAG_HOURS = [1, 4, 8, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]

IMPUTATION_COLS = ["temp", "humidity", "wind_speed", "visibility", "dew_point", "clouds_all", "feels_like"]
