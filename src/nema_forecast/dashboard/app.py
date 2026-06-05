"""NEMA Load Forecasting — Streamlit dashboard entry point.

Launch with::

    streamlit run src/nema_forecast/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from nema_forecast.dashboard.pages import (
    diagnostics,
    executive_summary,
    how_it_works,
    live_forecast,
    model_vs_iso,
)


def main() -> None:
    st.set_page_config(
        page_title="NEMA Load Forecast",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()

    # --- Sidebar navigation ---
    with st.sidebar:
        st.markdown(
            "<h2 style='margin-bottom:0'>⚡ NEMA Forecast</h2>"
            "<p style='color:#5D6D7E;margin-top:0'>New England Mass Boston</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Executive Summary",
                "Live Forecast",
                "Model vs ISO-NE",
                "Diagnostics",
                "How The Model Works",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Data: ISO-NE · OpenWeatherMap")
        st.caption("Model: Beacon v1.0")

    # --- Page dispatch ---
    if page == "Executive Summary":
        executive_summary.render()
    elif page == "Live Forecast":
        live_forecast.render()
    elif page == "Model vs ISO-NE":
        model_vs_iso.render()
    elif page == "Diagnostics":
        diagnostics.render()
    elif page == "How The Model Works":
        how_it_works.render()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Force a consistent light palette regardless of config.toml load or OS dark
           mode. Without this, a dark Streamlit base + the hard-coded light cards/sidebar
           below render as a "half dark, half white" UI. */
        :root { color-scheme: light; }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        .stApp,
        [data-testid="stMain"],
        [data-testid="stMain"] .block-container {
            background-color: #FAFCFE !important;
            color: #1C2833 !important;
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] span,
        [data-testid="stMain"] label,
        [data-testid="stMarkdownContainer"] {
            color: #1C2833 !important;
        }
        section[data-testid="stSidebar"] * { color: #1C2833 !important; }

        /* KPI metric cards */
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #E5E8E8;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetric"] label {
            color: #5D6D7E;
            font-size: 0.85rem;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* Sidebar tweaks */
        section[data-testid="stSidebar"] {
            background: #F4F6F7;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-size: 1.05rem;
            padding: 6px 0;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
        }

        /* Clean headers */
        h1, h2, h3 { font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
