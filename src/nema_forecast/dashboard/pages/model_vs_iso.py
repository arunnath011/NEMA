"""Page 3 — Model vs ISO-NE: deep comparison between Beacon and ISO-NE forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error

from nema_forecast.dashboard.components import (
    BLUE,
    GREEN,
    GREY,
    RED,
    bar_chart,
    dual_bar_chart,
    scatter_chart,
    timeseries_chart,
)
from nema_forecast.dashboard.live_data import build_recent_comparison
from nema_forecast.model.backtest import compute_hourly_metrics, compute_monthly_metrics

COMPARISON_DAYS = 30


def render() -> None:
    st.title("Beacon vs ISO-NE Forecast")
    st.markdown(
        f"Horizon-matched, side-by-side evaluation: **Beacon's day-ahead (24 h) forecast** "
        f"against ISO New England's day-ahead demand forecast for NEMA, over the last "
        f"**{COMPARISON_DAYS} days**. Both forecast the same hours 24 h ahead and use the same "
        "Open-Meteo weather forecast — an apples-to-apples comparison."
    )

    with st.spinner("Building live comparison from ISO-NE data …"):
        bt = build_recent_comparison(days=COMPARISON_DAYS)

    if bt.empty:
        st.warning(
            "No live comparison data available. This page needs ISO-NE Web Services "
            "credentials (ISO_NE_WS_USER / ISO_NE_WS_PASS) and the trained model."
        )
        return

    latest = bt["datetime"].max()
    st.caption(
        f"Data through **{latest:%b %d, %Y %H:%M}** · {len(bt):,} hours · "
        "actual = ISO-NE real-time demand, benchmark = ISO-NE day-ahead demand."
    )

    has_iso = "iso_forecast" in bt.columns and bt["iso_forecast"].notna().any()

    # ------------------------------------------------------------------
    # Date range selector
    # ------------------------------------------------------------------
    min_dt = bt["datetime"].min().date()
    max_dt = bt["datetime"].max().date()
    date_range = st.date_input("Date range", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        bt = bt[(bt["datetime"].dt.date >= date_range[0]) & (bt["datetime"].dt.date <= date_range[1])]

    tab_ts, tab_scatter, tab_error, tab_hourly, tab_extreme = st.tabs(
        ["Time Series", "Scatter Plots", "Error Distribution", "Hourly Breakdown", "Extreme Days"]
    )

    # ------------------------------------------------------------------
    # Tab 1: time series
    # ------------------------------------------------------------------
    with tab_ts:
        week_options = _build_week_options(bt)
        chosen_week = st.selectbox("Select week", week_options, index=len(week_options) - 1)
        week_df = bt[bt["datetime"].dt.isocalendar().week.astype(int) == int(chosen_week.split()[-1])]
        if week_df.empty:
            week_df = bt.tail(168)

        ts_cols: dict[str, str] = {"actual": "Actual", "catboost_pred": "Beacon"}
        if has_iso:
            ts_cols["iso_forecast"] = "ISO-NE"
        st.plotly_chart(timeseries_chart(week_df, ts_cols, ylabel="Load (MW)"), use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 2: scatter plots
    # ------------------------------------------------------------------
    with tab_scatter:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Beacon**")
            st.plotly_chart(
                scatter_chart(bt["actual"], bt["catboost_pred"], label="Beacon", colour=BLUE),
                use_container_width=True,
            )
        with c2:
            if has_iso:
                iso_valid = bt.dropna(subset=["iso_forecast"])
                st.markdown("**ISO-NE**")
                st.plotly_chart(
                    scatter_chart(iso_valid["actual"], iso_valid["iso_forecast"], label="ISO-NE", colour=GREEN),
                    use_container_width=True,
                )
            else:
                st.info("No ISO-NE forecast data available for scatter plot.")

    # ------------------------------------------------------------------
    # Tab 3: error distributions
    # ------------------------------------------------------------------
    with tab_error:
        fig = go.Figure()
        cat_err = bt["catboost_pred"] - bt["actual"]
        fig.add_trace(go.Histogram(x=cat_err, nbinsx=80, name="Beacon", marker_color=BLUE, opacity=0.7))
        if has_iso:
            iso_err = bt["iso_forecast"] - bt["actual"]
            fig.add_trace(go.Histogram(x=iso_err.dropna(), nbinsx=80, name="ISO-NE", marker_color=GREEN, opacity=0.6))
        fig.add_vline(x=0, line_dash="dash", line_color=RED, line_width=2)
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Forecast Error (MW)",
            yaxis_title="Count",
            template="plotly_white",
            legend={"orientation": "h", "y": 1.1},
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error statistics table
        stats_rows = [
            {
                "Model": "Beacon",
                "Mean Error": f"{cat_err.mean():.1f}",
                "Std Dev": f"{cat_err.std():.1f}",
                "Median": f"{cat_err.median():.1f}",
                "P5": f"{cat_err.quantile(0.05):.1f}",
                "P95": f"{cat_err.quantile(0.95):.1f}",
            }
        ]
        if has_iso:
            ie = bt["iso_forecast"] - bt["actual"]
            stats_rows.append(
                {
                    "Model": "ISO-NE",
                    "Mean Error": f"{ie.mean():.1f}",
                    "Std Dev": f"{ie.std():.1f}",
                    "Median": f"{ie.median():.1f}",
                    "P5": f"{ie.quantile(0.05):.1f}",
                    "P95": f"{ie.quantile(0.95):.1f}",
                }
            )
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Tab 4: hourly breakdown
    # ------------------------------------------------------------------
    with tab_hourly:
        hourly = compute_hourly_metrics(bt)
        x = hourly["hour"].tolist()
        y_cat = hourly.get("catboost_mae", pd.Series(dtype=float)).tolist()
        y_iso = hourly.get("iso_mae", pd.Series(dtype=float)).tolist()

        if has_iso and y_iso:
            st.plotly_chart(dual_bar_chart(x, y_cat, y_iso), use_container_width=True)
        else:
            st.plotly_chart(
                bar_chart(x, y_cat, title="MAE by Hour of Day", ylabel="MAE (MW)"), use_container_width=True
            )

        # Monthly
        monthly = compute_monthly_metrics(bt)
        if not monthly.empty:
            st.subheader("MAE by Month")
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            mx = [month_names[m - 1] for m in monthly["month"]]
            my_cat = monthly.get("catboost_mae", pd.Series(dtype=float)).tolist()
            my_iso = monthly.get("iso_mae", pd.Series(dtype=float)).tolist()
            if has_iso and my_iso:
                st.plotly_chart(dual_bar_chart(mx, my_cat, my_iso), use_container_width=True)
            else:
                st.plotly_chart(bar_chart(mx, my_cat, ylabel="MAE (MW)"), use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 5: extreme days
    # ------------------------------------------------------------------
    with tab_extreme:
        st.markdown("Performance on the **top 5 % peak-load** hours.")
        threshold = np.nanpercentile(bt["actual"], 95)
        extreme = bt[bt["actual"] >= threshold]
        normal = bt[bt["actual"] < threshold]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Threshold (95th pct)", f"{threshold:,.0f} MW")
        with c2:
            ext_mae = mean_absolute_error(extreme["actual"], extreme["catboost_pred"])
            st.metric("Extreme MAE", f"{ext_mae:.1f} MW")
        with c3:
            norm_mae = mean_absolute_error(normal["actual"], normal["catboost_pred"])
            st.metric("Normal MAE", f"{norm_mae:.1f} MW")

        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=normal["actual"],
                y=normal["catboost_pred"],
                mode="markers",
                marker={"size": 3, "color": BLUE, "opacity": 0.3},
                name="Normal",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=extreme["actual"],
                y=extreme["catboost_pred"],
                mode="markers",
                marker={"size": 5, "color": RED, "opacity": 0.7},
                name="Extreme (top 5%)",
            )
        )
        lo = min(bt["actual"].min(), bt["catboost_pred"].min())
        hi = max(bt["actual"].max(), bt["catboost_pred"].max())
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": GREY}, showlegend=False)
        )
        fig.update_layout(
            xaxis_title="Actual (MW)",
            yaxis_title="Predicted (MW)",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


def _build_week_options(df: pd.DataFrame) -> list[str]:
    weeks = df["datetime"].dt.isocalendar().week.astype(int).unique()
    return [f"Week {w}" for w in sorted(weeks)]
