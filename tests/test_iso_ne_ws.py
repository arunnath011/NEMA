"""Offline tests for the ISO-NE Web Services demand parser (no network)."""

from __future__ import annotations

import pandas as pd

from nema_forecast.data.iso_ne_ws import parse_demand

_RT_PAYLOAD = {
    "HourlyRtDemands": {
        "HourlyRtDemand": [
            {
                "BeginDate": "2025-05-30T00:00:00.000-04:00",
                "Load": 2510.5,
                "Location": {"@LocId": "4008", "$": "NEMASSBOST"},
            },
            {
                "BeginDate": "2025-05-30T01:00:00.000-04:00",
                "Load": 2480.0,
                "Location": {"@LocId": "4008", "$": "NEMASSBOST"},
            },
        ]
    }
}

_DA_PAYLOAD = {
    "HourlyDaDemands": {
        "HourlyDaDemand": [
            {
                "BeginDate": "2025-05-30T00:00:00.000-04:00",
                "Load": 2600.0,
                "Location": {"@LocId": "4008", "$": "NEMASSBOST"},
            }
        ]
    }
}

# Single-hour payloads come back as a bare object, not a list.
_RT_SINGLE = {
    "HourlyRtDemands": {
        "HourlyRtDemand": {
            "BeginDate": "2025-05-30T05:00:00.000-04:00",
            "Load": 2401.2,
            "Location": {"@LocId": "4008", "$": "NEMASSBOST"},
        }
    }
}


def test_parse_realtime_demand() -> None:
    df = parse_demand(_RT_PAYLOAD, wrapper="HourlyRtDemands", item="HourlyRtDemand", value_col="RTLO")
    assert list(df.columns) == ["datetime", "RTLO"]
    assert len(df) == 2
    # Eastern offset normalised to naive local wall-clock.
    assert df["datetime"].iloc[0] == pd.Timestamp("2025-05-30 00:00:00")
    assert df["RTLO"].iloc[0] == 2510.5
    assert df["datetime"].is_monotonic_increasing


def test_parse_dayahead_demand() -> None:
    df = parse_demand(_DA_PAYLOAD, wrapper="HourlyDaDemands", item="HourlyDaDemand", value_col="iso_forecast")
    assert list(df.columns) == ["datetime", "iso_forecast"]
    assert df["iso_forecast"].iloc[0] == 2600.0


def test_parse_single_hour_object() -> None:
    df = parse_demand(_RT_SINGLE, wrapper="HourlyRtDemands", item="HourlyRtDemand", value_col="RTLO")
    assert len(df) == 1
    assert df["RTLO"].iloc[0] == 2401.2


def test_parse_empty_payload() -> None:
    df = parse_demand({}, wrapper="HourlyRtDemands", item="HourlyRtDemand", value_col="RTLO")
    assert df.empty
    assert list(df.columns) == ["datetime", "RTLO"]

    df2 = parse_demand({"HourlyRtDemands": {}}, wrapper="HourlyRtDemands", item="HourlyRtDemand", value_col="RTLO")
    assert df2.empty
