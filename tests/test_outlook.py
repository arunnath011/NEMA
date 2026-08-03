"""Serving-path guard for the calendar-future Outlook model.

Exercises predict_outlook offline against synthetic load + weather so a future change that breaks
the feature-vector alignment (lag feats + target exog + lead_h) is caught. Skips if the Outlook
model artifact has not been trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nema_forecast.model.inference import load_outlook_model, predict_outlook


def _synthetic(hours: int = 320):
    idx = pd.date_range("2026-06-01", periods=hours, freq="h")
    daily = 300 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    load = pd.DataFrame({"datetime": idx, "RTLO": 2600 + daily})
    wx_idx = pd.date_range("2026-06-01", periods=hours + 240, freq="h")
    weather = pd.DataFrame(
        {
            "datetime": wx_idx,
            "temp": 72.0,
            "humidity": 60.0,
            "wind_speed": 6.0,
            "dew_point": 52.0,
            "clouds_all": 30.0,
            "feels_like": 72.0,
            "visibility": 10000.0,
        }
    )
    return load, weather


def test_predict_outlook_shape_and_range():
    if load_outlook_model()[0] is None:
        pytest.skip("Outlook model not trained (models/catboost_outlook.cbm missing)")
    load, weather = _synthetic()
    out = predict_outlook(load, weather)

    assert list(out.columns) == ["datetime", "forecast_mw", "lead_h"]
    assert not out.empty
    # Leads stay within the trained range; forecasts are physically plausible NEMA loads.
    assert out["lead_h"].min() >= 24
    assert out["lead_h"].max() <= 168
    assert out["forecast_mw"].between(500, 9000).all()
    # Targets are strictly after the last known load hour.
    assert out["datetime"].min() > load["datetime"].max()


def test_predict_outlook_empty_on_short_history():
    if load_outlook_model()[0] is None:
        pytest.skip("Outlook model not trained")
    load, weather = _synthetic(hours=50)  # < LOOKBACK
    out = predict_outlook(load, weather)
    assert out.empty
    assert list(out.columns) == ["datetime", "forecast_mw", "lead_h"]
