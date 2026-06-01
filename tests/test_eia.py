"""Offline tests for the EIA demand parser (no network)."""

from __future__ import annotations

import pandas as pd

from nema_forecast.data.eia import parse_eia_demand

_PAYLOAD = {
    "response": {
        "data": [
            {"period": "2025-05-30T04", "subba": "4008", "value": 2510, "value-units": "MWh"},
            {"period": "2025-05-30T05", "subba": "4008", "value": 2480, "value-units": "MWh"},
        ]
    }
}


def test_parse_eia_demand() -> None:
    df = parse_eia_demand(_PAYLOAD)
    assert list(df.columns) == ["datetime", "RTLO"]
    assert len(df) == 2
    # 04:00 UTC → 00:00 Eastern (EDT, -4) wall-clock, naive.
    assert df["datetime"].iloc[0] == pd.Timestamp("2025-05-30 00:00:00")
    assert df["RTLO"].iloc[0] == 2510
    assert df["datetime"].is_monotonic_increasing


def test_parse_eia_empty() -> None:
    assert parse_eia_demand({}).empty
    assert parse_eia_demand({"response": {"data": []}}).empty
    assert list(parse_eia_demand({}).columns) == ["datetime", "RTLO"]
