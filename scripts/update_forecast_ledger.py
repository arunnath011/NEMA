"""Update the locked forecast ledger — the daily job that makes Beacon's track record provable.

Two actions, in order:
  1. **Lock** a fresh day-ahead (24 h) forecast for the hours just after the latest available
     ISO-NE actual. Those hours' actuals are not yet published, so committing this row is a
     genuine ahead-of-time prediction — the git commit timestamp is the proof.
  2. **Score** any previously-locked rows whose actuals ISO-NE has since published.

Run from CI (GitHub Actions) or locally. Requires ISO-NE Web Services credentials
(``ISO_NE_WS_USER`` / ``ISO_NE_WS_PASS``) for the demand feeds and the trained model artifacts.

    python scripts/update_forecast_ledger.py
"""

from __future__ import annotations

import sys

import pandas as pd

from nema_forecast.config import ISO_NE_WS_PASS, ISO_NE_WS_USER, LEDGER_PATH, LOOKBACK
from nema_forecast.data.iso_ne_ws import fetch_dayahead_demand_recent, fetch_realtime_demand_recent
from nema_forecast.data.open_meteo import fetch_recent_weather
from nema_forecast.forecast_ledger import append_forecasts, load_ledger, save_ledger, score_pending, summarize
from nema_forecast.model.inference import load_model, predict_next_24h


def main() -> int:
    if not ISO_NE_WS_USER or not ISO_NE_WS_PASS:
        print(
            "ERROR: ISO_NE_WS_USER / ISO_NE_WS_PASS are not set. In GitHub, add them under "
            "Settings → Secrets and variables → Actions → New repository secret (they are the "
            "same ISO-NE Web Services credentials the app uses). Ledger unchanged.",
            file=sys.stderr,
        )
        return 1

    load = fetch_realtime_demand_recent(days_back=16)
    if load.empty or len(load) < LOOKBACK + 1:
        print(
            f"ERROR: not enough ISO-NE real-time demand ({len(load)} rows) — credentials may be "
            "wrong or ISO-NE may be briefly unavailable. Ledger unchanged.",
            file=sys.stderr,
        )
        return 1

    weather = fetch_recent_weather(past_days=92, forecast_days=16)
    origin = pd.to_datetime(load["datetime"]).max()

    # Beacon's genuine day-ahead forecast for the 24 h after the last actual (target-hour
    # weather + calendar). These target hours are not yet published by ISO-NE → a real forecast.
    fc = predict_next_24h(load, weather, model=load_model())
    if fc.empty:
        print("ERROR: forecast produced no rows; ledger unchanged.", file=sys.stderr)
        return 1

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

    ledger = load_ledger()
    before = len(ledger)
    ledger = append_forecasts(ledger, new_rows)
    iso = fetch_dayahead_demand_recent(days_back=24)
    ledger = score_pending(ledger, load, iso)
    save_ledger(ledger)

    s = summarize(ledger)
    print(
        f"Ledger {LEDGER_PATH}: {s['n_total']} rows (+{s['n_total'] - before} locked), "
        f"{s['n_scored']} scored, {s['n_pending']} pending. "
        f"Origin {origin} → forecast {target.min()}..{target.max()}."
    )
    if "mae" in s:
        iso_txt = f", ISO-NE {s['iso_mae']:.0f} MW" if "iso_mae" in s else ""
        print(f"Live track-record MAE so far: Beacon {s['mae']:.0f} MW{iso_txt} over {s['n_scored']} scored hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
