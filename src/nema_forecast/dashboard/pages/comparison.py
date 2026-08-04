"""Landing page — NEMA load forecasting with Beacon.

Leads with the **calendar-future Outlook** (what Beacon expects over the next ~4 days), then a
condensed strip of held-out-year accuracy as the proof that backs it up.
"""

from __future__ import annotations

import streamlit as st

from nema_forecast.dashboard.components import (
    horizon_accuracy_chart,
    load_horizon_mae,
    load_metrics,
)
from nema_forecast.dashboard.live_data import LiveDataError, build_recent_comparison
from nema_forecast.dashboard.outlook_view import render_outlook

# Kept short so the landing page loads fast (fewer live API calls + hindcast points).
COMPARISON_DAYS = 14


def render() -> None:
    st.title("NEMA Load Forecast")
    st.markdown(
        "**Beacon** forecasts electricity demand for the **NEMA (Northeast Massachusetts / "
        "Boston)** zone for the days ahead — a separate weather-aware model per hour, driven by "
        "the **forecasted weather** and calendar, the dominant drivers of demand. Here is what it "
        "expects next, and the track record that backs it up."
    )

    _hero_outlook()
    st.divider()
    _proof_strip()


# ---------------------------------------------------------------------------
# Hero — the calendar-future Outlook (what's coming)
# ---------------------------------------------------------------------------


def _hero_outlook() -> None:
    st.subheader("Beacon's forecast — the next 4 days")
    st.markdown(
        "ISO-NE publishes NEMA's real demand on a ~2-3 day lag, so the actuals below stop a few "
        "days back. Beacon's **Outlook** keeps going — a genuine forecast for hours whose actuals "
        "don't exist yet, out to ~4 days ahead."
    )
    try:
        with st.spinner("Loading live ISO-NE data and computing the forecast …"):
            bt = build_recent_comparison(days=COMPARISON_DAYS)
    except LiveDataError as exc:
        st.info(f"Live forecast temporarily unavailable — {exc} The proven accuracy below is unaffected.")
        return
    if bt.empty:
        st.info("Live forecast unavailable — no overlapping live hours yet. The proven accuracy below is unaffected.")
        return
    render_outlook(bt, hero=True, recent_days=7)


# ---------------------------------------------------------------------------
# Proof — condensed held-out-year accuracy (why you can trust it)
# ---------------------------------------------------------------------------


def _proof_strip() -> None:
    metrics = load_metrics()
    if not metrics:
        return

    st.subheader("Why you can trust it")
    n = metrics.get("test_samples", 0)
    st.markdown(
        f"Beacon was tested on a **strictly held-out year** (~{n:,} hours of 2025, none seen in "
        f"training): its day-ahead forecast lands within **{metrics.get('MAE_h24', 0):.0f} MW** of "
        f"real load on average and beats a naive one-step model by "
        f"**{metrics.get('avg_improvement_vs_single_pct', 0):.0f}%** across the horizon."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Day-ahead MAE (held-out year)", f"{metrics.get('MAE_h24', 0):.0f} MW")
    c2.metric("Fit quality (R²)", f"{metrics.get('R2', 0):.3f}")
    c3.metric("Better than naive baseline", f"{metrics.get('avg_improvement_vs_single_pct', 0):.0f}%")

    hm = load_horizon_mae()
    if hm:
        with st.expander("Accuracy by how far ahead we forecast"):
            st.plotly_chart(horizon_accuracy_chart(hm), use_container_width=True)
            st.caption(
                "Held-out error by horizon: Beacon stays flat as the horizon grows (each hour has "
                "its own weather-aware model), where a single one-step model rolled forward "
                "collapses at mid-day."
            )

    st.caption(
        "See **Live Track Record** for forecasts locked in git *before* the actuals were "
        "published, and **Model vs ISO-NE** for the full head-to-head comparison."
    )
