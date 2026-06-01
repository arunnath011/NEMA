"""Tests for data ingestion and preprocessing modules."""

from __future__ import annotations

import pandas as pd
import pytest

from nema_forecast.config import TRAIN_CUTOFF
from nema_forecast.data.preprocessing import (
    apply_imputation,
    clean_columns,
    compute_imputation_stats,
    temporal_split,
)


@pytest.fixture
def sample_load_df() -> pd.DataFrame:
    dates = pd.date_range("2024-12-30", periods=72, freq="h")
    return pd.DataFrame(
        {
            "datetime": dates,
            "RTLO": [2500 + i * 10 for i in range(72)],
            "sea_level": [None] * 72,
            "wind_gust": [None] * 72,
        }
    )


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    dates = pd.date_range("2024-12-30", periods=72, freq="h")
    return pd.DataFrame(
        {
            "datetime": dates,
            "temp": [35.0 + i * 0.1 for i in range(72)],
            "humidity": [60.0] * 72,
            "wind_speed": [5.0] * 72,
            "visibility": [10000] * 72,
        }
    )


def test_clean_columns_drops_high_missing(sample_load_df: pd.DataFrame) -> None:
    cleaned = clean_columns(sample_load_df)
    assert "sea_level" not in cleaned.columns
    assert "wind_gust" not in cleaned.columns
    assert "RTLO" in cleaned.columns


def test_temporal_split(sample_load_df: pd.DataFrame) -> None:
    train, test = temporal_split(sample_load_df, TRAIN_CUTOFF)
    assert len(train) > 0
    assert len(test) > 0
    assert train["datetime"].max() <= pd.to_datetime(TRAIN_CUTOFF)
    assert test["datetime"].min() > pd.to_datetime(TRAIN_CUTOFF)


def test_imputation_stats(sample_weather_df: pd.DataFrame) -> None:
    stats = compute_imputation_stats(sample_weather_df)
    assert "temp" in stats
    assert isinstance(stats["temp"], float)


def test_apply_imputation() -> None:
    df = pd.DataFrame({"temp": [30.0, None, 40.0], "humidity": [None, 60.0, None]})
    stats = {"temp": 35.0, "humidity": 55.0}
    result = apply_imputation(df, stats)
    assert result["temp"].isna().sum() == 0
    assert result["humidity"].isna().sum() == 0
