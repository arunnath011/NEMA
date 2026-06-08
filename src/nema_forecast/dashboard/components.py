"""Reusable dashboard UI components — KPI cards, chart helpers, data loaders."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nema_forecast.config import MODELS_DIR

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLUE = "#1B4F72"
GREEN = "#1E8449"
RED = "#C0392B"
GREY = "#7F8C8D"
LIGHT_BLUE = "#5DADE2"
LIGHT_GREEN = "#58D68D"

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_test_results() -> pd.DataFrame:
    path = MODELS_DIR / "test_results.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=300)
def load_backtest_results() -> pd.DataFrame:
    path = MODELS_DIR / "backtest_results.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    path = MODELS_DIR / "model_performance.json"
    if not path.exists():
        return {}
    data: dict = json.loads(path.read_text())
    return data


@st.cache_data(ttl=300)
def load_backtest_metrics() -> dict:
    path = MODELS_DIR / "backtest_metrics.json"
    if not path.exists():
        return {}
    data: dict = json.loads(path.read_text())
    return data


@st.cache_data(ttl=300)
def load_feature_importance() -> dict:
    path = MODELS_DIR / "feature_importance.json"
    if not path.exists():
        return {}
    data: dict = json.loads(path.read_text())
    return data


@st.cache_data(ttl=300)
def load_horizon_mae() -> dict:
    """Per-horizon MAE: direct multi-horizon models vs the single-model baseline."""
    path = MODELS_DIR / "horizon_mae.json"
    if not path.exists():
        return {}
    data: dict = json.loads(path.read_text())
    return data


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def timeseries_chart(
    df: pd.DataFrame,
    cols: dict[str, str],
    title: str = "",
    ylabel: str = "Load (MW)",
) -> go.Figure:
    """Build a multi-line time-series Plotly figure.

    *cols* maps column names → display labels, e.g. ``{"actual": "Actual", "catboost_pred": "Beacon"}``.
    """
    colours = [BLUE, GREEN, RED, GREY, LIGHT_BLUE]
    fig = go.Figure()
    for i, (col, label) in enumerate(cols.items()):
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df[col],
                mode="lines",
                name=label,
                line={"color": colours[i % len(colours)], "width": 2 if i == 0 else 1.5},
                opacity=1.0 if i == 0 else 0.85,
            )
        )
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
        height=420,
    )
    return fig


def scatter_chart(
    actual: pd.Series,
    predicted: pd.Series,
    label: str = "Predicted",
    colour: str = BLUE,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=actual,
            y=predicted,
            mode="markers",
            marker={"size": 3, "color": colour, "opacity": 0.4},
            name=label,
        )
    )
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": RED, "width": 1.5},
            name="Perfect",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis_title="Actual (MW)",
        yaxis_title="Predicted (MW)",
        template="plotly_white",
        height=400,
        margin={"t": 30, "b": 40, "l": 60, "r": 20},
    )
    return fig


def bar_chart(
    x: list,
    y: list,
    title: str = "",
    ylabel: str = "",
    colour: str = BLUE,
) -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colour))
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template="plotly_white",
        height=380,
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
    )
    return fig


def dual_bar_chart(
    x: list,
    y1: list,
    y2: list,
    label1: str = "Beacon",
    label2: str = "ISO-NE",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y1, name=label1, marker_color=BLUE))
    fig.add_trace(go.Bar(x=x, y=y2, name=label2, marker_color=GREEN))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        yaxis_title="MAE (MW)",
        height=400,
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
    )
    return fig


def horizon_accuracy_chart(hm: dict) -> go.Figure:
    """Per-horizon MAE: direct per-horizon Beacon vs the naive single-model baseline."""
    h, direct, single = hm["horizon"], hm["direct_mae"], hm["single_model_mae"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=h,
            y=single,
            mode="lines+markers",
            name="Old single model (rolled out)",
            line={"color": GREY, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=h,
            y=direct,
            mode="lines+markers",
            name="Beacon (per-horizon + weather)",
            line={"color": GREEN, "width": 3},
        )
    )
    fig.update_layout(
        xaxis_title="Forecast horizon (hours ahead)",
        yaxis_title="MAE (MW)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        height=400,
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
    )
    return fig


def kpi_delta(current: float, baseline: float) -> str:
    """Format a delta string for st.metric."""
    diff = current - baseline
    pct = diff / baseline * 100 if baseline else 0
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f} MW ({sign}{pct:.1f}%)"
