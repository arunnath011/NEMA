"""Page 1 — Executive Summary: live KPI cards, rolling performance, latest predictions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nema_forecast.dashboard.components import timeseries_chart
from nema_forecast.dashboard.live_data import build_recent_comparison
from nema_forecast.model.backtest import compute_metrics

COMPARISON_DAYS = 30


def render() -> None:
    st.title("Executive Summary")
    st.markdown(
        "High-level **Beacon** performance for the **NEMA (New England Mass Boston)** load "
        f"zone, computed live over the last **{COMPARISON_DAYS} days** of ISO-NE data."
    )

    with st.spinner("Building live performance summary from ISO-NE data …"):
        bt = build_recent_comparison(days=COMPARISON_DAYS)

    if bt.empty:
        st.warning(
            "No live data available. This page needs ISO-NE Web Services credentials "
            "(ISO_NE_WS_USER / ISO_NE_WS_PASS) and the trained model."
        )
        return

    metrics = compute_metrics(bt)
    cat = metrics.get("catboost", {})
    iso = metrics.get("iso", {})

    latest = bt["datetime"].max()
    st.caption(
        f"Data through **{latest:%b %d, %Y %H:%M}** · {len(bt):,} hours · "
        "actual = ISO-NE real-time demand, benchmark = ISO-NE day-ahead demand."
    )

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Beacon MAE",
            f"{cat.get('MAE', 0):.1f} MW",
            delta=f"{cat.get('MAE', 0) - iso['MAE']:.1f} MW vs ISO" if iso else None,
            delta_color="inverse",
        )
    with col2:
        st.metric(
            "Beacon MAPE",
            f"{cat.get('MAPE', 0):.2f}%",
            delta=f"{cat.get('MAPE', 0) - iso['MAPE']:.2f}% vs ISO" if iso else None,
            delta_color="inverse",
        )
    with col3:
        st.metric("Beacon R²", f"{cat.get('R2', 0):.4f}")
    with col4:
        if iso and iso.get("MAE"):
            improvement = (iso["MAE"] - cat.get("MAE", 0)) / iso["MAE"] * 100
            st.metric("Improvement over ISO", f"{improvement:.1f}%")
        else:
            st.metric("Hours Evaluated", f"{cat.get('n', len(bt)):,}")

    st.divider()

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    if iso:
        st.subheader("Model Comparison")
        comp_df = pd.DataFrame(
            {
                "Metric": ["MAE (MW)", "MAPE (%)", "R²", "Observations"],
                "Beacon": [
                    f"{cat.get('MAE', 0):.2f}",
                    f"{cat.get('MAPE', 0):.2f}",
                    f"{cat.get('R2', 0):.4f}",
                    f"{cat.get('n', '')}",
                ],
                "ISO-NE Forecast": [
                    f"{iso['MAE']:.2f}",
                    f"{iso['MAPE']:.2f}",
                    f"{iso['R2']:.4f}",
                    f"{iso.get('n', '')}",
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # ------------------------------------------------------------------
    # Rolling 7-day performance
    # ------------------------------------------------------------------
    st.subheader("Rolling 7-Day MAE")
    bt = bt.sort_values("datetime")
    bt["error_cat"] = (bt["actual"] - bt["catboost_pred"]).abs()
    bt["rolling_mae_cat"] = bt["error_cat"].rolling(168, min_periods=24).mean()

    cols_map = {"rolling_mae_cat": "Beacon (7-day rolling MAE)"}
    if "iso_forecast" in bt.columns and bt["iso_forecast"].notna().any():
        bt["error_iso"] = (bt["actual"] - bt["iso_forecast"]).abs()
        bt["rolling_mae_iso"] = bt["error_iso"].rolling(168, min_periods=24).mean()
        cols_map["rolling_mae_iso"] = "ISO-NE (7-day rolling MAE)"

    fig = timeseries_chart(bt, cols_map, ylabel="MAE (MW)")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Latest week sample
    # ------------------------------------------------------------------
    st.subheader("Latest Week — Actual vs Predicted")
    recent = bt.tail(168)
    ts_cols = {"actual": "Actual Load", "catboost_pred": "Beacon"}
    if "iso_forecast" in recent.columns and recent["iso_forecast"].notna().any():
        ts_cols["iso_forecast"] = "ISO-NE"
    fig2 = timeseries_chart(recent, ts_cols, ylabel="Load (MW)")
    st.plotly_chart(fig2, use_container_width=True)
