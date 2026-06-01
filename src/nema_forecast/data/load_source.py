"""Live load-source facade — prefer ISO-NE Web Services, fall back to EIA.

Lets the dashboard run with fresh NEMA demand from whichever source is configured:

1. **ISO-NE Web Services** (`realtimehourlydemand`) — the authoritative source, used when
   `ISO_NE_WS_USER` / `ISO_NE_WS_PASS` are set and the call succeeds.
2. **EIA Open Data** (`ISNE`/`4008`) — the no-wait fallback (free, instant API key), used
   when ISO-NE credentials are missing or the request fails.

Both return the same ``[datetime, RTLO]`` shape, so callers are source-agnostic. The active
source label is returned so the UI can show provenance.
"""

from __future__ import annotations

import logging

import pandas as pd

from nema_forecast.config import EIA_API_KEY, ISO_NE_WS_PASS, ISO_NE_WS_USER

logger = logging.getLogger(__name__)


def get_recent_demand(days_back: int = 10) -> tuple[pd.DataFrame, str]:
    """Return recent NEMA hourly demand and the name of the source that provided it.

    Returns ``(df, source)`` where *df* has columns ``[datetime, RTLO]`` and *source* is one
    of ``"ISO-NE Web Services"``, ``"EIA"``, or ``"none"`` (when no source yields data).
    """
    if ISO_NE_WS_USER and ISO_NE_WS_PASS:
        try:
            from nema_forecast.data.iso_ne_ws import fetch_realtime_demand_recent

            df = fetch_realtime_demand_recent(days_back=days_back)
            if not df.empty:
                return df, "ISO-NE Web Services"
            logger.warning("ISO-NE Web Services returned no data; trying EIA fallback")
        except Exception as exc:
            logger.warning("ISO-NE Web Services failed (%s); trying EIA fallback", exc)

    if EIA_API_KEY:
        try:
            from nema_forecast.data.eia import fetch_eia_demand_recent

            df = fetch_eia_demand_recent(days_back=days_back)
            if not df.empty:
                return df, "EIA"
            logger.warning("EIA returned no data")
        except Exception as exc:
            logger.warning("EIA fallback failed: %s", exc)

    return pd.DataFrame(columns=["datetime", "RTLO"]), "none"
