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
from nema_forecast.dashboard.live_data import LiveDataError, build_recent_comparison, get_outlook
from nema_forecast.model.backtest import compute_hourly_metrics, compute_monthly_metrics

COMPARISON_DAYS = 30
ISO_ORANGE = "#E67E22"
OUTLOOK_GREEN = "#16A085"  # Beacon's forward Outlook — distinct from the day-ahead green
OUTLOOK_SHADE = "rgba(22, 160, 133, 0.07)"


def _outlook_meta() -> dict:
    import json

    from nema_forecast.config import MODELS_DIR

    path = MODELS_DIR / "outlook_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _outlook_section(bt: pd.DataFrame) -> None:
    """Beacon's calendar-future Outlook, contrasted with the recent live day-ahead comparison."""
    st.subheader("Calendar-future Outlook — forecasting past the data lag")
    st.markdown(
        "ISO-NE publishes NEMA zonal actuals on a **~2-3 day lag**, so the horizon-matched "
        "comparison further down can only score *already-published* hours. Beacon's **Outlook** "
        "keeps going — a genuine forecast for the hours after the last actual, out to **~4 days "
        "ahead**, on dates whose actuals do not exist yet. Past the one-week lag window the "
        "forecast rides on the weather forecast and calendar, so its accuracy stays roughly flat."
    )

    outlook = get_outlook()
    if outlook.empty:
        st.info(
            "Outlook temporarily unavailable — it needs the live ISO-NE feed, weather, and the "
            "trained Outlook model. The comparison below is unaffected."
        )
        return

    scored = bt.dropna(subset=["actual"])
    last_actual = scored["datetime"].max() if not scored.empty else bt["datetime"].max()
    recent = bt[bt["datetime"] >= last_actual - pd.Timedelta(days=10)]

    peak = outlook.loc[outlook["forecast_mw"].idxmax()]
    heldout = _outlook_meta().get("heldout_by_lead", {})
    mae_d1 = heldout.get("96", {}).get("mae")
    mae_d4 = heldout.get("168", {}).get("mae")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted peak (next ~4 days)", f"{peak['forecast_mw']:,.0f} MW", help=f"at {peak['datetime']:%b %d, %H:%M}"
    )
    c2.metric("Outlook reaches", f"{outlook['datetime'].max():%b %d}")
    if mae_d1 and mae_d4:
        c3.metric(
            "Held-out error (+1d → +4d)", f"{mae_d1:.0f} → {mae_d4:.0f} MW", help="On the strictly held-out year."
        )

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
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Left of the marker: Beacon's and ISO-NE's day-ahead forecasts over hours ISO-NE has now "
        "published (scored against the real demand). Right of the marker: Beacon's Outlook for "
        "hours not yet published — a real ahead-of-time forecast, not a hindcast."
    )


def render() -> None:
    st.title("Beacon vs ISO-NE Forecast")
    st.markdown(
        f"Horizon-matched, side-by-side evaluation: **Beacon's day-ahead (24 h) forecast** "
        f"against ISO New England's day-ahead demand forecast for NEMA, over the last "
        f"**{COMPARISON_DAYS} days**. Both forecast the same hours 24 h ahead and use the same "
        "Open-Meteo weather forecast — an apples-to-apples comparison."
    )

    try:
        with st.spinner("Building live comparison from ISO-NE data …"):
            bt = build_recent_comparison(days=COMPARISON_DAYS)
    except LiveDataError as exc:
        st.warning(f"No live comparison data available — {exc} It should recover on the next refresh.")
        return

    if bt.empty:
        st.warning("No live comparison data available — no overlapping live hours to score yet.")
        return

    _outlook_section(bt)
    st.divider()

    st.subheader("Recent horizon-matched comparison (published hours)")
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
