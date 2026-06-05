"""Page 2 — Live Forecast: fetch current ISO-NE data + weather, run model, show results."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nema_forecast.config import LOOKBACK, MODELS_DIR
from nema_forecast.dashboard.components import BLUE, GREEN, GREY, RED, timeseries_chart

logger = logging.getLogger(__name__)


def render() -> None:
    st.title("Live Forecast")
    st.markdown(
        "Real-time data from **ISO-NE** and **OpenWeatherMap** — "
        "model generates a fresh 24-hour-ahead prediction from the latest available load."
    )

    # ------------------------------------------------------------------
    # Current weather
    # ------------------------------------------------------------------
    st.subheader("Current Weather — Boston")
    try:
        from nema_forecast.data.weather import fetch_current_weather

        weather = fetch_current_weather()
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Temperature", f"{weather.get('temp', 'N/A')}\u00b0F")
        with c2:
            st.metric("Feels Like", f"{weather.get('feels_like', 'N/A')}\u00b0F")
        with c3:
            st.metric("Humidity", f"{weather.get('humidity', 'N/A')}%")
        with c4:
            st.metric("Wind Speed", f"{weather.get('wind_speed', 'N/A')} mph")
        with c5:
            st.metric("Conditions", weather.get("weather_description", "N/A").title())
    except Exception as exc:
        st.info(f"Live weather unavailable: {exc}")
        weather = {}

    st.divider()

    # ------------------------------------------------------------------
    # Fetch recent load data from ISO-NE
    # ------------------------------------------------------------------
    st.subheader("Recent Load Data (Live from ISO-NE)")

    with st.spinner("Fetching latest NEMA load data …"):
        recent_load, source = _fetch_recent_load_cached()

    if recent_load.empty:
        st.error(
            "Could not fetch recent load data. Set ISO-NE Web Services credentials "
            "(ISO_NE_WS_USER / ISO_NE_WS_PASS) or an EIA API key (EIA_API_KEY)."
        )
        return

    latest_dt = recent_load["datetime"].max()
    earliest_dt = recent_load["datetime"].min()
    st.success(
        f"Loaded **{len(recent_load):,}** hourly observations from **{source}** — "
        f"**{earliest_dt:%b %d, %Y}** to **{latest_dt:%b %d, %Y %H:%M}**"
    )

    from datetime import datetime

    staleness = datetime.now() - pd.to_datetime(latest_dt)
    age_hours = staleness.total_seconds() / 3600
    st.caption(
        f"Load data as of **{latest_dt:%b %d, %Y %H:%M}** ({age_hours:.0f} h ago) — "
        f"source: {source}. Real-time hourly demand publishes with ~1 h latency."
    )

    # ------------------------------------------------------------------
    # Run inference on the latest window
    # ------------------------------------------------------------------
    st.subheader("24-Hour Forecast from Latest Data")

    forecast_df = _run_live_inference(recent_load, weather)

    if forecast_df is not None and not forecast_df.empty:
        # Show recent actuals + forecast
        recent_window = recent_load.tail(168).copy()
        recent_window = recent_window.rename(columns={"RTLO": "actual"})

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent_window["datetime"],
                y=recent_window["actual"],
                mode="lines",
                name="Actual Load",
                line={"color": BLUE, "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["datetime"],
                y=forecast_df["forecast_mw"],
                mode="lines+markers",
                name="CatBoost Forecast",
                line={"color": GREEN, "width": 2.5, "dash": "dash"},
                marker={"size": 5},
            )
        )
        # Overlay the ISO-NE day-ahead demand for the same window (the benchmark)
        try:
            iso_da = _fetch_dayahead_cached()
            if not iso_da.empty:
                iso_window = iso_da[
                    (iso_da["datetime"] >= forecast_df["datetime"].min())
                    & (iso_da["datetime"] <= forecast_df["datetime"].max())
                ]
                if not iso_window.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=iso_window["datetime"],
                            y=iso_window["iso_forecast"],
                            mode="lines",
                            name="ISO-NE Day-Ahead",
                            line={"color": GREY, "width": 2, "dash": "dot"},
                        )
                    )
        except Exception as exc:
            logger.info("ISO-NE day-ahead overlay unavailable: %s", exc)
        # Add a vertical line at the forecast start
        fig.add_shape(
            type="line",
            x0=str(latest_dt),
            x1=str(latest_dt),
            y0=0,
            y1=1,
            yref="paper",
            line={"dash": "dot", "color": GREY},
        )
        fig.add_annotation(
            x=str(latest_dt),
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yshift=10,
            font={"size": 11, "color": GREY},
        )
        fig.update_layout(
            yaxis_title="Load (MW)",
            template="plotly_white",
            legend={"orientation": "h", "y": 1.1},
            height=450,
            margin={"t": 40, "b": 40, "l": 60, "r": 20},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        with st.expander("Forecast Details"):
            display = forecast_df.copy()
            display["datetime"] = display["datetime"].dt.strftime("%b %d %H:%M")
            display["forecast_mw"] = display["forecast_mw"].round(1)
            st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not generate forecast — need at least 168 hours of recent data.")

    st.divider()

    # ------------------------------------------------------------------
    # Last 7 days actual load
    # ------------------------------------------------------------------
    st.subheader("Last 7 Days — Actual NEMA Load")
    last_week = recent_load.tail(168).copy()
    last_week = last_week.rename(columns={"RTLO": "actual"})
    fig2 = timeseries_chart(last_week, {"actual": "Actual Load (MW)"}, ylabel="Load (MW)")
    st.plotly_chart(fig2, use_container_width=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    rtlo = last_week["actual"]
    with c1:
        st.metric("7-Day Mean", f"{rtlo.mean():,.0f} MW")
    with c2:
        st.metric("7-Day Peak", f"{rtlo.max():,.0f} MW")
    with c3:
        st.metric("7-Day Min", f"{rtlo.min():,.0f} MW")
    with c4:
        st.metric("Std Dev", f"{rtlo.std():,.0f} MW")

    # ------------------------------------------------------------------
    # 5-Day weather forecast
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("5-Day Weather Forecast — Boston")
    try:
        from nema_forecast.data.weather import fetch_weather_forecast

        forecast_weather = fetch_weather_forecast()
        if not forecast_weather.empty:
            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=forecast_weather["datetime"],
                    y=forecast_weather["temp"],
                    mode="lines",
                    name="Temperature (\u00b0F)",
                    line={"color": RED},
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=forecast_weather["datetime"],
                    y=forecast_weather["humidity"],
                    mode="lines",
                    name="Humidity (%)",
                    line={"color": BLUE, "dash": "dash"},
                    yaxis="y2",
                )
            )
            fig3.update_layout(
                yaxis={"title": "Temperature (\u00b0F)"},
                yaxis2={"title": "Humidity (%)", "overlaying": "y", "side": "right"},
                template="plotly_white",
                legend={"orientation": "h", "y": 1.1},
                height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)
    except Exception as exc:
        st.info(f"Weather forecast unavailable: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_recent_load_cached() -> tuple[pd.DataFrame, str]:
    """Fetch recent NEMA demand via the source facade (cached 1 h in Streamlit).

    Prefers ISO-NE Web Services, falls back to EIA. Demand publishes with ~1-2 h latency,
    so 10 days is plenty of fresh history to seed the 168 h lookback window. Returns
    ``(df, source_label)``.
    """
    from nema_forecast.data.load_source import get_recent_demand

    # ISO-NE realtimehourlydemand lags ~4 days, so request a wider window to ensure the
    # 168 h (7-day) lookback is comfortably covered after the most recent empty days.
    return get_recent_demand(days_back=14)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_dayahead_cached() -> pd.DataFrame:
    """Fetch recent ISO-NE day-ahead demand (the benchmark) — cached 1 h."""
    from nema_forecast.data.iso_ne_ws import fetch_dayahead_demand_recent

    return fetch_dayahead_demand_recent(days_back=3)


def _run_live_inference(recent_load: pd.DataFrame, weather: dict) -> pd.DataFrame | None:
    """Run the recursive 24 h forecast on the latest real-time demand window.

    Weather over the forecast horizon comes from the OpenWeatherMap 5-day forecast; the
    *weather* dict (current observation) is unused here but kept for signature stability.
    """
    from nema_forecast.model.inference import load_model, predict_next_24h

    model_path = MODELS_DIR / "catboost_model.cbm"
    stats_path = MODELS_DIR / "imputation_stats.json"
    if not model_path.exists() or not stats_path.exists():
        return None

    if len(recent_load) < LOOKBACK:
        return None

    try:
        model = load_model(model_path)

        # Forecast weather covers the horizon hours; fall back to an empty frame so
        # imputation fills weather from training medians if the API is unavailable.
        try:
            from nema_forecast.data.weather import fetch_weather_forecast

            recent_weather = fetch_weather_forecast()
        except Exception:
            recent_weather = pd.DataFrame()

        return predict_next_24h(recent_load, recent_weather, model=model)

    except Exception as exc:
        logger.error("Live inference failed: %s", exc, exc_info=True)
        return None
