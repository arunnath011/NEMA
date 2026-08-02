"""Update the locked forecast ledger — manual / local runner.

Locks a fresh day-ahead forecast for the hours just after the latest ISO-NE actual (their
actuals are not published yet, so committing the row is a genuine ahead-of-time prediction) and
scores any previously-locked rows whose actuals have since been published, then writes the CSV.

The deployed app does this on its own via the GitHub API (ISO-NE blocks GitHub Actions IPs, so
it can't run in CI). This script is for a manual local refresh or a machine ISO-NE permits:

    python scripts/update_forecast_ledger.py
"""

from __future__ import annotations

import sys

from nema_forecast.config import ISO_NE_WS_PASS, ISO_NE_WS_USER, LEDGER_PATH
from nema_forecast.forecast_ledger import save_ledger, summarize
from nema_forecast.ledger_update import refresh_ledger


def main() -> int:
    if not ISO_NE_WS_USER or not ISO_NE_WS_PASS:
        print(
            "ERROR: ISO_NE_WS_USER / ISO_NE_WS_PASS are not set (put them in .env locally). " "Ledger unchanged.",
            file=sys.stderr,
        )
        return 1

    ledger, changed = refresh_ledger()
    if ledger.empty:
        print("ERROR: no ISO-NE demand available; ledger unchanged.", file=sys.stderr)
        return 1

    save_ledger(ledger)
    s = summarize(ledger)
    verb = "updated" if changed else "unchanged"
    print(f"Ledger {LEDGER_PATH} {verb}: {s['n_total']} rows, {s['n_scored']} scored, {s['n_pending']} pending.")
    if "mae" in s:
        iso_txt = f", ISO-NE {s['iso_mae']:.0f} MW" if "iso_mae" in s else ""
        print(f"Live track-record MAE so far: Beacon {s['mae']:.0f} MW{iso_txt} over {s['n_scored']} scored hours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
