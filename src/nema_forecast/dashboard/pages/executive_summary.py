"""Page 1 — Executive Summary: KPI cards, rolling performance, latest predictions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nema_forecast.dashboard.components import (
    load_backtest_metrics,
    load_backtest_results,
    load_metrics,
    load_test_results,
    timeseries_chart,
)


def render() -> None:
    st.title("Executive Summary")
    st.markdown("High-level model performance for the **NEMA (New England Mass Boston)** load zone.")

    # ------------------------------------------------------------------
    # KPI row
    # ------------------------------------------------------------------
    metrics = load_metrics()
    bt_metrics = load_backtest_metrics()

    if not metrics:
        st.warning("No model performance data found. Run `python -m nema_forecast.model.train` first.")
        return

    cat = bt_metrics.get("catboost", metrics)
    iso = bt_metrics.get("iso", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "CatBoost MAE",
            f"{cat.get('MAE', metrics.get('MAE', 0)):.1f} MW",
            delta=f"{cat.get('MAE', 0) - iso.get('MAE', cat.get('MAE', 0)):.1f} MW vs ISO" if iso else None,
            delta_color="inverse",
        )
    with col2:
        st.metric(
            "CatBoost MAPE",
            f"{cat.get('MAPE', metrics.get('MAPE', 0)):.2f}%",
            delta=f"{cat.get('MAPE', 0) - iso.get('MAPE', cat.get('MAPE', 0)):.2f}% vs ISO" if iso else None,
            delta_color="inverse",
        )
    with col3:
        st.metric("CatBoost R\u00b2", f"{cat.get('R2', metrics.get('R2', 0)):.4f}")
    with col4:
        if iso:
            improvement = (iso["MAE"] - cat.get("MAE", metrics["MAE"])) / iso["MAE"] * 100
            st.metric("Improvement over ISO", f"{improvement:.1f}%")
        else:
            st.metric("Test Samples", f"{metrics.get('test_samples', 'N/A'):,}")

    st.divider()

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    if iso:
        st.subheader("Model Comparison")
        comp_df = pd.DataFrame(
            {
                "Metric": ["MAE (MW)", "MAPE (%)", "R\u00b2", "Observations"],
                "CatBoost": [
                    f"{cat.get('MAE', metrics['MAE']):.2f}",
                    f"{cat.get('MAPE', metrics['MAPE']):.2f}",
                    f"{cat.get('R2', metrics['R2']):.4f}",
                    f"{cat.get('n', metrics.get('test_samples', ''))}",
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
    bt = load_backtest_results()
    if bt.empty:
        bt = load_test_results()

    if bt.empty:
        st.info("No backtest data available yet.")
        return

    st.subheader("Rolling 7-Day MAE")
    bt = bt.sort_values("datetime")
    bt["error_cat"] = (bt["actual"] - bt["catboost_pred"]).abs()
    bt["rolling_mae_cat"] = bt["error_cat"].rolling(168, min_periods=24).mean()

    cols_map = {"rolling_mae_cat": "CatBoost (7-day rolling MAE)"}

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
    ts_cols = {"actual": "Actual Load", "catboost_pred": "CatBoost"}
    if "iso_forecast" in recent.columns and recent["iso_forecast"].notna().any():
        ts_cols["iso_forecast"] = "ISO-NE"
    fig2 = timeseries_chart(recent, ts_cols, ylabel="Load (MW)")
    st.plotly_chart(fig2, use_container_width=True)
