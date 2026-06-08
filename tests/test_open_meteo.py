"""Offline tests for the Open-Meteo weather parser (no network)."""

from __future__ import annotations

import pandas as pd

from nema_forecast.data.open_meteo import _parse

_PAYLOAD = {
    "hourly": {
        "time": ["2026-05-30T00:00", "2026-05-30T01:00"],
        "temperature_2m": [55.4, 54.1],
        "relative_humidity_2m": [70, 72],
        "wind_speed_10m": [6.1, 5.5],
        "dew_point_2m": [46.0, 45.2],
        "cloud_cover": [40, 55],
        "apparent_temperature": [54.0, 52.8],
        "visibility": [24000.0, None],  # archive may omit visibility
    }
}


def test_parse_maps_to_model_columns() -> None:
    df = _parse(_PAYLOAD)
    assert list(df.columns) == [
        "datetime",
        "temp",
        "humidity",
        "wind_speed",
        "dew_point",
        "clouds_all",
        "feels_like",
        "visibility",
    ]
    assert len(df) == 2
    assert df["temp"].iloc[0] == 55.4
    assert df["clouds_all"].iloc[1] == 55
    assert df["feels_like"].iloc[0] == 54.0
    # missing visibility filled with the constant default
    assert df["visibility"].iloc[1] == 10000.0
    assert df["datetime"].iloc[0] == pd.Timestamp("2026-05-30 00:00:00")


def test_parse_empty() -> None:
    assert _parse({}).empty
    assert _parse({"hourly": {"time": []}}).empty
