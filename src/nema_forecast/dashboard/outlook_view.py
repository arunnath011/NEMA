"""Shared Calendar-future Outlook view — rendered on both the landing hero and the Model-vs-ISO
tab, so the two never drift. Callers add their own subheading/intro, then call ``render_outlook``.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nema_forecast.config import MODELS_DIR
from nema_forecast.dashboard.components import BLUE, GREEN, GREY
from nema_forecast.dashboard.live_data import get_outlook

ISO_ORANGE = "#E67E22"
OUTLOOK_GREEN = "#16A085"  # Beacon's forward Outlook — distinct from the day-ahead green
OUTLOOK_SHADE = "rgba(22, 160, 133, 0.07)"


def _heldout_by_lead() -> dict:
    path = MODELS_DIR / "outlook_meta.json"
    return json.loads(path.read_text()).get("heldout_by_lead", {}) if path.exists() else {}


def render_outlook(bt: pd.DataFrame, *, hero: bool = False, recent_days: int = 10) -> bool:
    """Render the Outlook callouts + chart from the live comparison frame *bt*.

    ``hero`` adds a big plain-language peak sentence (for the landing). Returns False (and shows a
    small notice) if the Outlook is unavailable, so callers can decide what else to render.
    """
    outlook = get_outlook()
    if outlook.empty:
        st.info(
            "Live forecast is warming up — it needs the ISO-NE feed, weather, and the Outlook "
            "model. Try refreshing in a moment."
        )
        return False

    scored = bt.dropna(subset=["actual"])
    last_actual = scored["datetime"].max() if not scored.empty else bt["datetime"].max()
    recent = bt[bt["datetime"] >= last_actual - pd.Timedelta(days=recent_days)]

    peak = outlook.loc[outlook["forecast_mw"].idxmax()]
    heldout = _heldout_by_lead()
    mae_d1 = heldout.get("96", {}).get("mae")
    mae_d4 = heldout.get("168", {}).get("mae")

    if hero:
        st.markdown(
            f"### Beacon expects NEMA to peak at **~{peak['forecast_mw']:,.0f} MW** "
            f"around **{peak['datetime']:%A %b %d, %-I %p}**."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted peak (next ~4 days)", f"{peak['forecast_mw']:,.0f} MW", help=f"at {peak['datetime']:%b %d, %H:%M}"
    )
    c2.metric("Outlook reaches", f"{outlook['datetime'].max():%b %d}")
    if mae_d1 and mae_d4:
        c3.metric(
            "Held-out error (+1d → +4d)", f"{mae_d1:.0f} → {mae_d4:.0f} MW", help="On the strictly held-out year."
        )

    st.plotly_chart(_figure(recent, outlook, last_actual), use_container_width=True)
    st.caption(
        "Left of the marker: Beacon's and ISO-NE's day-ahead forecasts over hours ISO-NE has now "
        "published (scored against real demand). Right of the marker: Beacon's Outlook for hours "
        "not yet published — a real ahead-of-time forecast, not a hindcast."
    )
    return True


def _figure(recent: pd.DataFrame, outlook: pd.DataFrame, last_actual) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["datetime"],
            y=recent["actual"],
            mode="lines",
            name="Actual (real demand)",
            line={"color": BLUE, "width": 2.4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=recent["datetime"],
            y=recent["catboost_pred"],
            mode="lines",
            name="Beacon (day-ahead, scored)",
            line={"color": GREEN, "width": 1.8},
        )
    )
    if "iso_forecast" in recent.columns and recent["iso_forecast"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=recent["datetime"],
                y=recent["iso_forecast"],
                mode="lines",
                name="ISO-NE (day-ahead)",
                line={"color": ISO_ORANGE, "width": 1.5, "dash": "dot"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=outlook["datetime"],
            y=outlook["forecast_mw"],
            mode="lines",
            name="Beacon Outlook (forecast)",
            line={"color": OUTLOOK_GREEN, "width": 2.6, "dash": "dash"},
        )
    )
    fig.add_vrect(x0=last_actual, x1=outlook["datetime"].max(), fillcolor=OUTLOOK_SHADE, line_width=0)
    fig.add_vline(x=last_actual, line_width=1.5, line_dash="dot", line_color=GREY)
    fig.add_annotation(
        x=last_actual,
        y=1.02,
        yref="paper",
        showarrow=False,
        text="latest ISO-NE actual",
        font={"color": GREY, "size": 11},
        xanchor="right",
    )
    fig.update_layout(
        yaxis_title="Load (MW)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        height=460,
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
    )
    return fig
