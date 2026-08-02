"""Tests for the locked forecast ledger — integrity rules must hold: forecasts are immutable,
appends never overwrite, and scoring only fills matured rows."""

from __future__ import annotations

import pandas as pd

from nema_forecast import forecast_ledger as fl


def _rows(targets, made_at, forecast=100.0):
    origin = targets[0] - pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "target_datetime": targets,
            "forecast_mw": [forecast + i for i in range(len(targets))],
            "forecast_made_at": made_at,
            "origin_datetime": origin,
            "horizon_h": [(t - origin) / pd.Timedelta(hours=1) for t in targets],
        }
    )


def test_append_locks_new_hours_and_never_overwrites():
    t = pd.date_range("2026-07-30", periods=4, freq="h")
    first = fl.append_forecasts(fl._coerce(pd.DataFrame()), _rows(t, pd.Timestamp("2026-07-29T12:00Z"), 100.0))
    assert len(first) == 4

    # A second run re-forecasts the same hours (different values, later timestamp) + one new hour.
    t2 = pd.date_range("2026-07-30 02:00", periods=4, freq="h")  # overlaps 02:00, 03:00; adds 04:00, 05:00
    second = fl.append_forecasts(first, _rows(t2, pd.Timestamp("2026-07-30T12:00Z"), 999.0))

    assert len(second) == 6  # only the two genuinely-new hours are added
    # Overlapping hours keep their ORIGINAL forecast + made_at (immutability).
    orig = first.set_index("target_datetime")["forecast_mw"]
    merged = second.set_index("target_datetime")["forecast_mw"]
    for hour in t[2:]:
        assert merged[hour] == orig[hour]
    assert second["forecast_made_at"].nunique() == 2


def test_score_pending_fills_only_matured_rows():
    t = pd.date_range("2026-07-30", periods=4, freq="h")
    ledger = fl.append_forecasts(fl._coerce(pd.DataFrame()), _rows(t, pd.Timestamp("2026-07-29T12:00Z"), 100.0))

    # Actuals published for only the first two hours.
    actuals = pd.DataFrame({"datetime": t[:2], "RTLO": [111.0, 222.0]})
    iso = pd.DataFrame({"datetime": t[:2], "iso_forecast": [110.0, 220.0]})
    scored = fl.score_pending(ledger, actuals, iso)

    assert scored["actual_mw"].notna().sum() == 2
    assert scored.iloc[0]["actual_mw"] == 111.0
    assert scored.iloc[2]["actual_mw"] != scored.iloc[2]["actual_mw"]  # still NaN (not published)
    assert scored["scored_at"].notna().sum() == 2
    # Forecasts are untouched by scoring.
    assert list(scored["forecast_mw"]) == list(ledger["forecast_mw"])


def test_score_is_idempotent_and_forecast_immutable():
    t = pd.date_range("2026-07-30", periods=2, freq="h")
    ledger = fl.append_forecasts(fl._coerce(pd.DataFrame()), _rows(t, pd.Timestamp("2026-07-29T12:00Z"), 100.0))
    actuals = pd.DataFrame({"datetime": t, "RTLO": [111.0, 222.0]})

    once = fl.score_pending(ledger, actuals)
    # A later run reports different "actuals" for the same hours; already-scored rows must not change.
    twice = fl.score_pending(once, pd.DataFrame({"datetime": t, "RTLO": [999.0, 999.0]}))
    assert list(twice["actual_mw"]) == [111.0, 222.0]
