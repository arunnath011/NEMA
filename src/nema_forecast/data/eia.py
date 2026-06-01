"""EIA Open Data API client — fallback live source for NEMA hourly demand.

The EIA publishes hourly demand by ISO subregion. NEMA is a first-class subregion
(``parent=ISNE``, ``subba=4008`` → NEMASSBOST, legacy series ``EBA.ISNE-4008.D.H``) with
~1-2 h latency. A free API key is issued instantly, so this is the no-wait fallback used
when ISO-NE Web Services credentials are unavailable.

Like the ISO-NE Web Services client, the demand series is returned under the column name
``RTLO`` (real-time metered demand in MW) so the rest of the pipeline needs no changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from nema_forecast.config import (
    EIA_API_KEY,
    EIA_BASE_URL,
    EIA_ISNE_PARENT,
    EIA_NEMA_SUBBA,
)

logger = logging.getLogger(__name__)

# EIA hourly ``period`` values are UTC ("YYYY-MM-DDTHH"); convert to local wall-clock,
# naive, to match the calendar features (and the ISO-NE Web Services client).
_ISO_NE_TZ = "America/New_York"


def fetch_eia_demand_recent(days_back: int = 10) -> pd.DataFrame:
    """Fetch recent NEMA hourly demand from the EIA API. Returns ``[datetime, RTLO]``."""
    if not EIA_API_KEY:
        raise RuntimeError(
            "EIA_API_KEY missing. Get a free key instantly at "
            "https://www.eia.gov/opendata/register.php and set it in .env / Streamlit secrets."
        )

    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[parent][]": EIA_ISNE_PARENT,
        "facets[subba][]": EIA_NEMA_SUBBA,
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    try:
        resp = requests.get(EIA_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("EIA request failed: %s", exc)
        return pd.DataFrame(columns=["datetime", "RTLO"])
    except ValueError as exc:
        logger.warning("EIA returned non-JSON: %s", exc)
        return pd.DataFrame(columns=["datetime", "RTLO"])

    return parse_eia_demand(payload)


def parse_eia_demand(payload: dict) -> pd.DataFrame:
    """Parse an EIA region-sub-ba-data response into ``[datetime, RTLO]`` (pure, offline)."""
    empty = pd.DataFrame(columns=["datetime", "RTLO"])
    rows = (payload or {}).get("response", {}).get("data")
    if not rows:
        return empty

    records = []
    for r in rows:
        if not isinstance(r, dict) or "period" not in r or "value" not in r:
            continue
        records.append({"datetime": r["period"], "RTLO": r["value"]})
    if not records:
        return empty

    df = pd.DataFrame.from_records(records)
    # period is UTC hour; localise to Eastern wall-clock, naive.
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = dt.dt.tz_convert(_ISO_NE_TZ).dt.tz_localize(None)
    df["RTLO"] = pd.to_numeric(df["RTLO"], errors="coerce")
    df = df.dropna(subset=["datetime", "RTLO"]).drop_duplicates("datetime", keep="last")
    return df.sort_values("datetime").reset_index(drop=True)
