"""Landing page — NEMA day-ahead load forecasting with Beacon.

Leads with Beacon's proven long-run (held-out full-year) accuracy, then shows the recent
live forecast against the real demand and ISO New England's operational day-ahead forecast.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from nema_forecast.dashboard.components import (
    BLUE,
    GREEN,
    horizon_accuracy_chart,
    load_horizon_mae,
    load_metrics,
)
from nema_forecast.dashboard.live_data import LiveDataError, build_recent_comparison

# Kept short so the landing page loads fast (fewer live API calls + hindcast points).
COMPARISON_DAYS = 14
ISO_ORANGE = "#E67E22"

_WINDOWS = {"Last 7 days": 168, "Last 14 days": 336}


def render() -> None:
    st.title("NEMA Day-Ahead Load Forecast")
    st.markdown(
        "**Beacon** forecasts electricity demand for the **NEMA (Northeast Massachusetts / "
        "Boston)** zone a full day ahead. It uses a separate gradient-boosting model for each "
        "of the next 24 hours, conditioned on a week of recent load and the **forecasted "
        "weather** at the target hour — the dominant driver of tomorrow's demand. Below: how "
        "accurate Beacon has been over a full held-out year, and how it is tracking right now "
        "against the real demand and ISO New England's own day-ahead forecast."
    )

    _long_run_panel()
    st.divider()
    _recent_live_section()


# ---------------------------------------------------------------------------
# Long-run accuracy (held-out full year) — the headline evidence
# ---------------------------------------------------------------------------


def _long_run_panel() -> None:
    metrics = load_metrics()
    if not metrics:
        return

    st.subheader("Accuracy proven over a full held-out year")
    n = metrics.get("test_samples", 0)
    st.markdown(
        f"Beacon was tested on a **strictly held-out year** — roughly {n:,} hours spanning "
        "winter, spring, summer, and fall of 2025, none of it seen during training. Across "
        f"that year its **day-ahead (24-hour) forecast** lands within **{metrics.get('MAE_h24', 0):.0f} MW** "
        "of the real load on average — and error grows only "
        f"**{metrics.get('horizon_degradation', 0):.2f}×** from one hour ahead to a full day "
        f"ahead, versus a naive one-step model that it beats by **{metrics.get('avg_improvement_vs_single_pct', 0):.0f}%** "
        "across the horizon. A short recent window (further down) reflects just one season and "
        "is noisier; **this full-year result is the fuller measure of Beacon's accuracy.**"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Day-ahead MAE (held-out year)", f"{metrics.get('MAE_h24', 0):.0f} MW")
    c2.metric("Fit quality (R²)", f"{metrics.get('R2', 0):.3f}")
    c3.metric("Better than naive baseline", f"{metrics.get('avg_improvement_vs_single_pct', 0):.0f}%")

    hm = load_horizon_mae()
    if hm:
        st.plotly_chart(horizon_accuracy_chart(hm), use_container_width=True)
        st.caption(
            "Forecast error by how far ahead we predict, on the held-out year. Beacon stays "
            "flat as the horizon grows (each hour has its own weather-aware model), where a "
            "single one-step model rolled forward collapses at mid-day. Measured with observed "
            "weather; the live day-ahead below additionally depends on the weather forecast."
        )


# ---------------------------------------------------------------------------
# Recent live — Real vs Beacon vs ISO-NE
# ---------------------------------------------------------------------------


def _recent_live_section() -> None:
    st.subheader("Recent live forecast — Real vs Beacon vs ISO-NE")
    st.markdown(
        "The most recent weeks of live NEMA demand, with Beacon's and ISO New England's "
        "day-ahead forecasts overlaid — both at the same 24-hour horizon, same weather. This is "
        "recomputed on demand over already-published data; for forecasts **locked in git before "
        "the actuals were published**, see the **Live Track Record** page."
    )

    try:
        with st.spinner("Loading live ISO-NE data and computing forecasts …"):
            bt = build_recent_comparison(days=COMPARISON_DAYS)
    except LiveDataError as exc:
        st.info(
            f"Live comparison temporarily unavailable — {exc} The held-out full-year results "
            "above are unaffected; the live view should recover on the next refresh."
        )
        return

    if bt.empty:
        st.info("Live comparison unavailable — no overlapping live hours to score yet.")
        return

    latest = bt["datetime"].max()
    st.caption(
        f"Live data through **{latest:%b %d, %Y %H:%M}** · {len(bt):,} hours · "
        "actual = ISO-NE real-time demand · forecasts at 24 h ahead."
    )

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
        height=460,
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
