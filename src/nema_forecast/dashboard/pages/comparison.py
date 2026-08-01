"""Landing page — Forecast Comparison: Actual (real) vs Beacon vs ISO-NE, day-ahead.

The single clear message of the dashboard: how the model's day-ahead forecast tracks the
real load, side by side with ISO New England's operational day-ahead forecast.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from nema_forecast.dashboard.components import BLUE, GREEN
from nema_forecast.dashboard.live_data import build_recent_comparison
from nema_forecast.model.backtest import compute_metrics

COMPARISON_DAYS = 30
ISO_ORANGE = "#E67E22"

_WINDOWS = {"Last 7 days": 168, "Last 14 days": 336, "Last 30 days": 720}


def render() -> None:
    st.title("Real vs Beacon vs ISO-NE")
    st.markdown(
        "Day-ahead (24 h) load forecast for the **NEMA** zone: the model's forecast "
        "(**Beacon**) and ISO New England's operational forecast, both against the "
        "**real** metered demand. Horizon-matched, using the same Open-Meteo weather."
    )

    with st.spinner("Loading live ISO-NE data and computing forecasts …"):
        bt = build_recent_comparison(days=COMPARISON_DAYS)

    if bt.empty:
        st.warning(
            "No live data available. This page needs ISO-NE Web Services credentials "
            "(ISO_NE_WS_USER / ISO_NE_WS_PASS)."
        )
        return

    m = compute_metrics(bt)
    beacon, iso = m.get("catboost", {}), m.get("iso", {})
    latest = bt["datetime"].max()

    # ------------------------------------------------------------------
    # Compact accuracy header — Beacon vs ISO-NE
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Beacon MAE",
        f"{beacon.get('MAE', 0):.1f} MW",
        delta=f"{beacon.get('MAE', 0) - iso['MAE']:.1f} vs ISO" if iso else None,
        delta_color="inverse",
    )
    c2.metric("ISO-NE MAE", f"{iso.get('MAE', 0):.1f} MW" if iso else "—")
    c3.metric(
        "Beacon MAPE",
        f"{beacon.get('MAPE', 0):.2f}%",
        delta=f"{beacon.get('MAPE', 0) - iso['MAPE']:.2f} vs ISO" if iso else None,
        delta_color="inverse",
    )
    c4.metric("ISO-NE MAPE", f"{iso.get('MAPE', 0):.2f}%" if iso else "—")

    st.caption(
        f"Data through **{latest:%b %d, %Y %H:%M}** · {len(bt):,} hours evaluated · "
        "actual = ISO-NE real-time demand · both forecasts at 24 h ahead."
    )

    # ------------------------------------------------------------------
    # Hero time-series — Actual vs Beacon vs ISO-NE
    # ------------------------------------------------------------------
    choice = st.radio("Window", list(_WINDOWS), index=1, horizontal=True, label_visibility="collapsed")
    win = bt.tail(_WINDOWS[choice])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=win["datetime"],
            y=win["actual"],
            mode="lines",
            name="Actual (real demand)",
            line={"color": BLUE, "width": 2.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=win["datetime"],
            y=win["catboost_pred"],
            mode="lines",
            name="Beacon (day-ahead)",
            line={"color": GREEN, "width": 2},
        )
    )
    if "iso_forecast" in win.columns and win["iso_forecast"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=win["datetime"],
                y=win["iso_forecast"],
                mode="lines",
                name="ISO-NE (day-ahead)",
                line={"color": ISO_ORANGE, "width": 1.8, "dash": "dot"},
            )
        )
    fig.update_layout(
        yaxis_title="Load (MW)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
        height=480,
        margin={"t": 30, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show data table"):
        tbl = win.rename(columns={"actual": "Actual", "catboost_pred": "Beacon", "iso_forecast": "ISO-NE"}).copy()
        tbl["datetime"] = tbl["datetime"].dt.strftime("%b %d %H:%M")
        for col in ("Actual", "Beacon", "ISO-NE"):
            if col in tbl.columns:
                tbl[col] = tbl[col].round(0)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
