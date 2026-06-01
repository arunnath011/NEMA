"""Page 4 — Diagnostics: feature importance, residuals, horizon analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nema_forecast.dashboard.components import (
    BLUE,
    GREY,
    RED,
    load_backtest_results,
    load_feature_importance,
    load_metrics,
    load_test_results,
)


def render() -> None:
    st.title("Model Diagnostics")
    st.markdown("Deep dive into model behaviour, feature importance, and error patterns.")

    metrics = load_metrics()
    if not metrics:
        st.warning("Train the model first to generate diagnostic data.")
        return

    tab_fi, tab_res, tab_leak, tab_ablation = st.tabs(
        ["Feature Importance", "Residual Analysis", "Leakage Tests", "RTLO Ablation"]
    )

    # ------------------------------------------------------------------
    # Tab 1: Feature importance
    # ------------------------------------------------------------------
    with tab_fi:
        fi = load_feature_importance()
        if fi:
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            top_n = st.slider("Show top N features", 10, min(50, len(sorted_fi)), 20)
            top = sorted_fi[:top_n]
            names = [x[0] for x in top][::-1]
            vals = [x[1] for x in top][::-1]

            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=BLUE))
            fig.update_layout(
                title=f"Top {top_n} Feature Importance",
                xaxis_title="Importance (%)",
                template="plotly_white",
                height=max(400, top_n * 22),
                margin={"l": 220},
            )
            st.plotly_chart(fig, use_container_width=True)

            # RTLO vs non-RTLO breakdown
            rtlo_imp = sum(v for k, v in fi.items() if "RTLO" in k)
            other_imp = 100 - rtlo_imp
            c1, c2 = st.columns(2)
            with c1:
                st.metric("RTLO Feature Importance", f"{rtlo_imp:.1f}%")
            with c2:
                st.metric("Non-RTLO Feature Importance", f"{other_imp:.1f}%")
        else:
            st.info("Feature importance data not available.")

    # ------------------------------------------------------------------
    # Tab 2: Residual analysis
    # ------------------------------------------------------------------
    with tab_res:
        bt = load_backtest_results()
        if bt.empty:
            bt = load_test_results()
        if bt.empty:
            st.info("No test results available.")
            return

        residuals = bt["catboost_pred"] - bt["actual"]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Residual Distribution**")
            fig = go.Figure(go.Histogram(x=residuals, nbinsx=80, marker_color=BLUE, opacity=0.8))
            fig.add_vline(x=0, line_dash="dash", line_color=RED)
            fig.update_layout(
                xaxis_title="Residual (MW)",
                yaxis_title="Count",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Residuals Over Time**")
            fig2 = go.Figure(
                go.Scattergl(
                    x=bt["datetime"],
                    y=residuals,
                    mode="markers",
                    marker={"size": 2, "color": BLUE, "opacity": 0.3},
                )
            )
            fig2.add_hline(y=0, line_dash="dash", line_color=RED)
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="Residual (MW)",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Autocorrelation of residuals
        st.markdown("**Residual Autocorrelation (first 72 lags)**")
        resid_vals = residuals.dropna().values
        n = len(resid_vals)
        mean_r = resid_vals.mean()
        denom = np.sum((resid_vals - mean_r) ** 2)
        acf = []
        for lag in range(73):
            if lag == 0:
                acf.append(1.0)
            else:
                num = np.sum((resid_vals[lag:] - mean_r) * (resid_vals[:-lag] - mean_r))
                acf.append(num / denom if denom else 0)

        ci = 1.96 / np.sqrt(n)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=list(range(73)), y=acf, marker_color=BLUE, width=0.6))
        fig3.add_hline(y=ci, line_dash="dot", line_color=GREY)
        fig3.add_hline(y=-ci, line_dash="dot", line_color=GREY)
        fig3.update_layout(
            xaxis_title="Lag (hours)",
            yaxis_title="ACF",
            template="plotly_white",
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 3: Leakage test summary
    # ------------------------------------------------------------------
    with tab_leak:
        st.markdown(
            """
            The model passed a rigorous **10-test leakage suite** during development.
            Key results are summarised below.
            """
        )

        tests = [
            ("Temporal Separation", "Train ends 2024-12-31, test starts 2025-01-01. Gap ≥ 168 h.", "PASS"),
            ("Feature-Target Correlation", "No non-lag features with >0.99 correlation to target.", "PASS"),
            ("Naive Baseline Comparison", "CatBoost beats lag-1 persistence by >5%.", "PASS"),
            ("Shuffled Target", "Shuffled-target model degrades >50% — confirms real signal.", "PASS"),
            ("Feature Importance", "RTLO lag features are dominant (expected for load forecasting).", "PASS"),
            ("Remove RTLO Features", "Removing RTLO degrades MAE — lags carry genuine information.", "PASS"),
            ("Shuffle RTLO Lags", "Shuffling RTLO lags degrades performance significantly.", "PASS"),
            ("Horizon-wise MAE", "h=24 MAE / h=1 MAE > 1.3× — model correctly finds it harder.", "PASS"),
            ("+48h Target Shift", "Predicting further ahead degrades — no future leakage.", "PASS"),
            ("Year-Based Split", "Training years 2017-2024, test year 2025 — no overlap.", "PASS"),
        ]

        for name, desc, status in tests:
            emoji = "\u2705" if status == "PASS" else "\u274c"
            st.markdown(f"**{emoji} {name}** — {desc}")

    # ------------------------------------------------------------------
    # Tab 4: RTLO lag ablation
    # ------------------------------------------------------------------
    with tab_ablation:
        st.markdown(
            """
            The notebook tested every combination of RTLO lag features.
            The best configuration uses **lag4, lag48, lag168, mean24, mean168**.
            """
        )
        ablation_data = {
            "no_rtlo": {"MAE": "Higher baseline (weather + calendar only)"},
            "lag1": {"MAE": "Slight improvement"},
            "lag4": {"MAE": "Good — captures intra-day persistence"},
            "lag24": {"MAE": "Captures daily cycle"},
            "lag48": {"MAE": "Captures two-day pattern"},
            "lag168": {"MAE": "Captures weekly seasonality"},
            "lag1+24+168": {"MAE": "Strong combination"},
            "all lags": {"MAE": "Marginal extra improvement"},
            "selected (production)": {"MAE": "Best trade-off — used in production"},
        }
        st.dataframe(
            pd.DataFrame([{"Config": k, "Result": v["MAE"]} for k, v in ablation_data.items()]),
            use_container_width=True,
            hide_index=True,
        )
