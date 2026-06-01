"""Tests for feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nema_forecast.features.engineering import (
    add_calendar_features,
    add_holiday_features,
    add_temperature_features,
    create_sequences,
    engineer_features,
    extract_lag_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.date_range("2024-07-01", periods=500, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "datetime": dates,
            "RTLO": 2500 + rng.normal(0, 200, 500),
            "temp": 60 + rng.normal(0, 15, 500),
            "humidity": 50 + rng.normal(0, 10, 500),
            "wind_speed": 5 + rng.normal(0, 2, 500),
            "visibility": 10000 + rng.normal(0, 1000, 500),
            "clouds_all": rng.integers(0, 100, 500).astype(float),
        }
    )


def test_calendar_features(sample_df: pd.DataFrame) -> None:
    result = add_calendar_features(sample_df)
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "dow_sin" in result.columns
    assert "is_weekend" in result.columns
    assert result["hour_sin"].between(-1, 1).all()


def test_holiday_features(sample_df: pd.DataFrame) -> None:
    result = add_holiday_features(sample_df)
    assert "is_us_holiday" in result.columns
    july4 = result[pd.to_datetime(result["datetime"]).dt.date == pd.to_datetime("2024-07-04").date()]
    assert july4["is_us_holiday"].iloc[0] == 1


def test_temperature_features(sample_df: pd.DataFrame) -> None:
    result = add_temperature_features(sample_df)
    assert "HDD" in result.columns
    assert "CDD" in result.columns
    assert "temp_heating_sigmoid" in result.columns
    assert (result["HDD"] >= 0).all()
    assert (result["CDD"] >= 0).all()


def test_engineer_features(sample_df: pd.DataFrame) -> None:
    result = engineer_features(sample_df)
    assert "datetime" in result.columns
    assert "RTLO" in result.columns
    assert "hour_sin" in result.columns
    assert len(result) <= len(sample_df)


def test_create_sequences(sample_df: pd.DataFrame) -> None:
    feat = engineer_features(sample_df)
    X, Y, cols, rtlo_idx = create_sequences(feat, lookback=48, horizon=6)
    assert X.ndim == 3
    assert Y.ndim == 2
    assert X.shape[1] == 48
    assert Y.shape[1] == 6
    assert cols[rtlo_idx] == "RTLO"


def test_extract_lag_features(sample_df: pd.DataFrame) -> None:
    feat = engineer_features(sample_df)
    X, _Y, cols, rtlo_idx = create_sequences(feat, lookback=168, horizon=24)
    X_gb, names = extract_lag_features(X, cols, rtlo_idx)
    assert X_gb.ndim == 2
    assert X_gb.shape[0] == X.shape[0]
    assert len(names) == X_gb.shape[1]
    rtlo_names = [n for n in names if n.startswith("RTLO_")]
    assert len(rtlo_names) <= 5  # only whitelisted RTLO features
