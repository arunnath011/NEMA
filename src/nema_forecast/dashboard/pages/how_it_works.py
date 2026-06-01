"""Page 5 — How The Model Works: detailed technical documentation of the NEMA forecasting pipeline."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from nema_forecast.dashboard.components import BLUE, GREEN, RED


def render() -> None:
    st.title("How The Model Works")
    st.markdown(
        "A comprehensive technical guide to the NEMA load forecasting pipeline — "
        "from raw data to 24-hour-ahead predictions."
    )

    tab_overview, tab_data, tab_features, tab_model, tab_leakage, tab_performance, tab_production = st.tabs(
        [
            "Overview",
            "Data Sources",
            "Feature Engineering",
            "Model Architecture",
            "Leakage Prevention",
            "Performance Analysis",
            "Production Notes",
        ]
    )

    # ==================================================================
    # OVERVIEW
    # ==================================================================
    with tab_overview:
        st.header("Problem Statement")
        st.markdown(
            """
            **NEMA (Northeast Massachusetts and Boston)** is one of eight electric load zones
            in the ISO New England wholesale electricity market.  Accurate short-term load
            forecasting (1–24 hours ahead) is critical for:

            - **Grid reliability** — ensuring sufficient generation is committed to serve demand.
            - **Market efficiency** — day-ahead and real-time energy prices depend on demand forecasts.
            - **Cost optimisation** — better forecasts reduce the need for expensive peaking generation.

            ISO New England publishes an official **three-day reliability region demand forecast**
            that grid operators rely on.  This project builds a **CatBoost gradient-boosting model**
            that uses historical load patterns and weather data to produce more accurate hourly
            forecasts for the NEMA zone.
            """
        )

        st.header("Pipeline At-a-Glance")
        st.markdown(_PIPELINE_HTML, unsafe_allow_html=True)

    # ==================================================================
    # DATA SOURCES
    # ==================================================================
    with tab_data:
        st.header("ISO-NE Load Data")
        st.markdown(
            """
            **Source:** [ISO New England — Hourly Wholesale Load Cost Reports](https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/whlsecost-hourly-nemassbost)

            The pipeline downloads monthly CSV files directly from ISO-NE.  Each file contains
            hourly rows with the columns:

            | Column | Description |
            |--------|-------------|
            | `RTLO` | **Real-Time Load Obligation** — the actual hourly demand in MW.  This is our **target variable**. |
            | `RTLMP` | Real-time locational marginal price |
            | `Capacity`, `Regulation`, … | Various wholesale cost components |
            | `LocalDate`, `LocalHour` | Timestamp (hour-ending convention, 1–24) |

            **Training period:** March 2017 — December 2024 (~68,000 hourly observations).

            **Test period:** January — November 2025 (~8,000 hourly observations).

            The temporal split at Dec 31 2024 ensures **strict out-of-sample evaluation** with
            no future data contamination.
            """
        )

        st.header("Weather Data")
        st.markdown(
            """
            **Source:** [OpenWeatherMap API](https://openweathermap.org/api) — Boston Logan International Airport
            (42.37°N, 71.01°W).

            Weather is fetched in **imperial units** (°F for temperature) at hourly granularity.

            | Feature | Correlation with Load | Role |
            |---------|----------------------|------|
            | `temp` | Strong (U-shaped) | Primary driver — heating and cooling demand |
            | `humidity` | Moderate | Increases perceived heat → AC load |
            | `wind_speed` | Weak-moderate | Wind chill in winter |
            | `visibility` | Weak | Proxy for storm conditions |
            | `clouds_all` | Weak | Cloud cover percentage |

            **PCMCI causal analysis** (see Feature Engineering tab) confirmed that `temp`,
            `humidity`, `wind_speed`, and `visibility` have statistically significant causal
            links to load at lags of 1–5 hours.
            """
        )

        st.header("ISO Three-Day Forecast (Benchmark)")
        st.markdown(
            """
            ISO-NE publishes a **Three-Day Reliability Region Demand Forecast** daily.
            This is the official forecast that grid operators use.  We compare our model
            against this benchmark on matched hours to demonstrate value.

            The forecast is extracted from daily CSV files filtering for region
            `.Z.NEMASSBOST`.  When multiple forecast vintages exist for the same hour,
            we use the **most recent** (closest to delivery).
            """
        )

    # ==================================================================
    # FEATURE ENGINEERING
    # ==================================================================
    with tab_features:
        st.header("Calendar Features")
        st.markdown(
            """
            Electricity demand follows strong **diurnal** (24h), **weekly** (weekday vs weekend),
            and **seasonal** (summer cooling, winter heating) cycles.

            Rather than using raw hour/day/month integers (which imply an artificial ordering —
            e.g., hour 23 is not "far" from hour 0), we use **cyclic encoding**:

            ```
            hour_sin = sin(2π · hour / 24)    hour_cos = cos(2π · hour / 24)
            dow_sin  = sin(2π · dow  / 7 )    dow_cos  = cos(2π · dow  / 7 )
            month_sin = sin(2π · month / 12)  month_cos = cos(2π · month / 12)
            ```

            This ensures that 11 PM and midnight are close in feature space.

            Additionally: `is_weekend` (binary) and `is_us_holiday` (US federal holidays).
            """
        )

        st.header("Temperature Features")
        st.markdown(
            """
            Load has a **U-shaped relationship** with temperature: demand is high when it's
            very cold (heating) or very hot (cooling), and lowest around 65°F.

            We capture this with multiple representations:
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **Linear degree-day features:**
                - `HDD` = max(65 − temp, 0)  — Heating Degree Days
                - `CDD` = max(temp − 65, 0)  — Cooling Degree Days

                **Polynomial features:**
                - `temp_sq`, `HDD_sq`, `CDD_sq` — capture non-linear extremes
                """
            )

        with col2:
            st.markdown(
                """
                **Sigmoidal features** (smooth S-curve transitions):
                - `temp_heating_sigmoid` — high when cold (center=18°C)
                - `temp_cooling_sigmoid` — high when hot (center=22°C)
                - `temp_thermal_stress` — combined extremity measure
                """
            )

        # Plot sigmoidal transforms
        temp_range = np.linspace(-10, 105, 200)
        heating = 1 - 1 / (1 + np.exp(-0.3 * (temp_range - 18)))
        cooling = 1 / (1 + np.exp(-0.3 * (temp_range - 22)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temp_range, y=heating, name="Heating Sigmoid", line={"color": BLUE}))
        fig.add_trace(go.Scatter(x=temp_range, y=cooling, name="Cooling Sigmoid", line={"color": RED}))
        fig.add_trace(
            go.Scatter(x=temp_range, y=heating + cooling, name="Thermal Stress", line={"color": GREEN, "dash": "dash"})
        )
        fig.update_layout(
            xaxis_title="Temperature (°F)",
            yaxis_title="Feature Value",
            template="plotly_white",
            height=350,
            title="Sigmoidal Temperature Transforms",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.header("Lag Features (168-Hour Lookback)")
        st.markdown(
            """
            The model uses a **168-hour (one week) lookback window** — enough to capture the
            full weekly demand cycle.  From this window, we extract:

            | Lag Feature | Hours Back | Rationale |
            |-------------|-----------|-----------|
            | `RTLO_lag4` | 4h | Recent intra-day trend |
            | `RTLO_lag48` | 48h | Two-day persistence |
            | `RTLO_lag168` | 168h | Same hour, same day last week |
            | `RTLO_mean24` | Rolling 24h mean | Daily average level |
            | `RTLO_mean168` | Rolling 168h mean | Weekly average level |

            Weather and calendar features also get lagged at [1, 4, 8, 24, 48, 168] hours.

            **Ablation study results:** Removing `RTLO_lag168` (weekly lag) caused the largest
            single-feature degradation, confirming that weekly seasonality is the most
            important autoregressive signal.
            """
        )

        st.header("PCMCI Causal Feature Selection")
        st.markdown(
            """
            We used **PCMCI** (Peter and Clark Momentary Conditional Independence) —
            a causal discovery algorithm from the `tigramite` library — to identify which
            weather variables have **genuine causal effects** on load (not just correlation).

            PCMCI tests conditional independence at multiple lags (1–5 hours) and controls
            for confounders.  Results confirmed significant causal links from:

            - **Temperature** → Load (lags 1–5h, p < 0.001)
            - **Humidity** → Load (lags 1–3h, p < 0.01)
            - **Wind speed** → Load (lags 1–2h, p < 0.05)
            - **Visibility** → Load (lag 1h, p < 0.05)

            This justifies including these four weather variables as model inputs while
            excluding others (e.g., `pressure`, `wind_deg`) that showed correlation but
            not causation.
            """
        )

    # ==================================================================
    # MODEL ARCHITECTURE
    # ==================================================================
    with tab_model:
        st.header("Why CatBoost?")
        st.markdown(
            """
            **CatBoost** (Categorical Boosting) is a gradient-boosted decision tree library
            developed by Yandex.  We chose it for NEMA load forecasting because:

            1. **Handles missing values natively** — our weather data has gaps; CatBoost
               uses optimal split conditions for NaN without requiring imputation heuristics.
            2. **Ordered boosting** — reduces prediction shift (a subtle form of target leakage
               inherent in standard gradient boosting).
            3. **Oblivious trees** — all leaves at the same depth use the same split features,
               making the model more regularised and faster at inference.
            4. **Strong out-of-the-box performance** — competitive with XGBoost/LightGBM with
               less hyperparameter sensitivity.
            """
        )

        st.header("Hyperparameters")
        params = {
            "iterations": ("1,000", "Maximum boosting rounds (early-stopped at best validation score)"),
            "learning_rate": ("0.05", "Conservative rate for stable convergence"),
            "depth": ("8", "Tree depth — captures complex feature interactions"),
            "loss_function": ("MAE", "Mean Absolute Error — robust to outliers in load data"),
            "early_stopping_rounds": ("50", "Stop if validation MAE doesn't improve for 50 rounds"),
            "random_seed": ("42", "Reproducibility"),
        }
        st.table(
            {
                "Parameter": list(params.keys()),
                "Value": [v[0] for v in params.values()],
                "Rationale": [v[1] for v in params.values()],
            }
        )

        st.header("Training / Validation / Test Split")
        st.markdown(
            """
            | Split | Period | Rows | Purpose |
            |-------|--------|------|---------|
            | Training | Mar 2017 – Oct 2024 | ~58,400 | Model fitting |
            | Validation | Nov – Dec 2024 | ~10,300 | Early stopping + hyperparameter selection |
            | Test | Jan – Nov 2025 | ~8,000 | Final evaluation (never seen during training) |

            The validation set is the **last 15%** of the training period — this maintains
            temporal ordering and prevents future-data leakage.
            """
        )

        st.header("Input / Output")
        st.markdown(
            """
            **Input:** A feature vector constructed from the 168-hour lookback window:
            - 5 RTLO lag/mean features
            - Weather lags at 6 time offsets × ~10 weather/calendar features
            - Calendar features at 6 time offsets
            - Total: ~80–100 features after the RTLO whitelist filter

            **Output:** Single scalar — predicted RTLO (MW) at horizon h=1
            (one hour ahead).

            For h=2 through h=24, the production model would ideally train 24 separate
            horizon-specific models.  The current v1.0 uses the h=1 model as a baseline
            for all horizons (the dashboard shows actual h=1 performance).
            """
        )

    # ==================================================================
    # LEAKAGE PREVENTION
    # ==================================================================
    with tab_leakage:
        st.header("Why Leakage Matters")
        st.markdown(
            """
            Data leakage is the #1 cause of models that look great in the lab but fail in
            production.  In time-series forecasting, leakage typically occurs when:

            1. Future values of the target (or correlated features) sneak into training rows.
            2. Test-period data influences imputation, scaling, or feature selection.
            3. The model memorises training patterns that don't generalise.

            We implemented a **10-test leakage suite** to catch all of these.
            """
        )

        st.header("5 Diagnostic Tests")
        tests = [
            (
                "1. Temporal Separation",
                "Verified that the last training timestamp (Dec 31 2024 23:00) is at least "
                "168 hours before the first test sample used for evaluation.  This ensures "
                "the lookback window never reaches into the training set.",
            ),
            (
                "2. Feature–Target Correlation",
                "Scanned all features for >0.99 correlation with the target.  Any non-lag "
                "feature with near-perfect correlation would indicate a target proxy leaking "
                "through.  All high correlations were from legitimate RTLO lags.",
            ),
            (
                "3. Naive Baseline Comparison",
                "The model must beat a lag-1 persistence baseline (ŷ(t) = y(t−1)) by at "
                "least 5%.  If it can't, the features aren't adding real information.",
            ),
            (
                "4. Shuffled Target Test",
                "We shuffled the training targets randomly and retrained.  The resulting "
                "model's MAE degraded by >50%, confirming the model is learning genuine "
                "patterns, not noise.",
            ),
            (
                "5. Feature Importance Audit",
                "Inspected the top features by CatBoost importance.  RTLO lag features "
                "dominated (~60–70% of total importance), which is expected and healthy "
                "for an autoregressive load model.",
            ),
        ]
        for title, desc in tests:
            with st.expander(title, expanded=False):
                st.markdown(desc)

        st.header("5 Forensic Tests")
        forensic = [
            (
                "6. Remove RTLO Features",
                "Retrained without any RTLO lag features.  MAE increased significantly, "
                "proving the lags carry genuine predictive value (not leakage).",
            ),
            (
                "7. Shuffle RTLO Lags",
                "Randomly permuted RTLO lag columns while keeping weather/calendar intact.  "
                "Large MAE degradation confirms the lag values (not just the lag positions) matter.",
            ),
            (
                "8. Horizon-wise MAE",
                "Trained separate models for h=1 through h=24.  MAE increased monotonically "
                "with horizon (h=24 MAE / h=1 MAE > 1.3×).  If there were leakage, all "
                "horizons would perform equally well.",
            ),
            (
                "9. +48h Target Shift",
                "Predicting 48 hours ahead produced worse MAE than 1 hour ahead.  This is "
                "the expected degradation pattern for a legitimate forecasting model.",
            ),
            (
                "10. Year-Based Analysis",
                "Training years (2017–2024) and test years (2025) have zero overlap.  "
                "The model can't memorise specific dates/events from the test period.",
            ),
        ]
        for title, desc in forensic:
            with st.expander(title, expanded=False):
                st.markdown(desc)

        st.header("CI Gate")
        st.markdown(
            """
            As a **continuous integration gate**, we enforce:

            > **h=24 MAE / h=1 MAE ≥ 1.3**

            If this ratio drops below 1.3, it signals that the model is too accurate at
            long horizons — a hallmark of future data leaking into features.  This check
            runs automatically before any model is deployed.
            """
        )

    # ==================================================================
    # PERFORMANCE ANALYSIS
    # ==================================================================
    with tab_performance:
        st.header("How We Beat the ISO-NE Forecast")
        st.markdown(
            """
            The CatBoost model consistently outperforms the official ISO-NE three-day
            forecast on the NEMA zone.  The primary reasons:

            **1. Autoregressive lag features capture recent trends the ISO misses.**

            The ISO forecast is generated once or twice daily and doesn't update hour-by-hour.
            Our model uses the most recent 168 hours of actual load as features, capturing
            real-time trends (e.g., a cold snap driving up demand).

            **2. Non-linear weather transforms.**

            The ISO likely uses a regression-based approach with limited non-linear terms.
            Our HDD/CDD polynomials and sigmoidal transforms better capture the U-shaped
            temperature–load relationship, especially at extremes.

            **3. Granular holiday handling.**

            We include not just federal holidays but also MA state holidays and the days
            immediately before/after holidays (when load patterns shift).
            """
        )

        st.header("Where the Model Struggles")
        st.markdown(
            """
            **1. Extreme heat events (top 5% peak-load hours)**

            MAE increases ~2× on peak days.  These events are rare in the training data,
            making it hard for the model to generalise.  The ISO-NE forecast is sometimes
            more conservative (over-predicts), which is actually safer for grid operations.

            **2. Holiday weekends**

            Demand patterns during long weekends (Thanksgiving, July 4th) differ from
            typical weekends, and we have few examples per holiday.

            **3. Sudden weather shifts**

            When temperature changes rapidly (>15°F in 6 hours), the lag features
            take several hours to "catch up," causing temporarily elevated errors.
            """
        )

        st.header("Error by Temperature Band")
        st.markdown(
            """
            | Temperature Range | Typical MAE | Notes |
            |-------------------|-------------|-------|
            | < 20°F | ~130 MW | Extreme cold — heating demand highly variable |
            | 20–50°F | ~90 MW | Mild cold — most predictable regime |
            | 50–70°F | ~80 MW | Comfortable range — lowest load and lowest error |
            | 70–85°F | ~100 MW | Moderate cooling — non-linear AC uptake |
            | > 85°F | ~140 MW | Extreme heat — peak demand, highest error |
            """
        )

    # ==================================================================
    # PRODUCTION NOTES
    # ==================================================================
    with tab_production:
        st.header("Data Freshness Requirements")
        st.markdown(
            """
            | Data Source | Update Frequency | Latency | Impact if Stale |
            |-------------|-----------------|---------|-----------------|
            | RTLO (load) | Hourly | ~1h delay | Critical — lag features degrade rapidly |
            | Weather (current) | Every 10 min | Real-time | Moderate — temp changes slowly |
            | Weather (forecast) | Every 3h | 0–3h | Low — used for forward-looking only |
            | ISO 3-day forecast | Daily | 12–24h | Low — benchmark comparison only |
            """
        )

        st.header("Model Retraining Cadence")
        st.markdown(
            """
            **Recommended: Monthly retraining** with expanding window.

            - Load patterns shift with seasons, economic activity, and infrastructure changes.
            - Monthly retraining ensures the model stays calibrated.
            - Each retrain takes ~5 minutes on a modern CPU (CatBoost is fast).
            - The CI gate (h=24/h=1 MAE ratio) must pass before deployment.
            """
        )

        st.header("Fallback Strategy")
        st.markdown(
            """
            If the CatBoost model fails (API down, corrupt data, etc.):

            1. **Primary fallback:** Use the ISO-NE three-day forecast.
            2. **Secondary fallback:** Seasonal naïve baseline (same hour, same day last week).
            3. **Emergency fallback:** Historical average by hour × day-of-week.

            The dashboard automatically detects when CatBoost predictions are unavailable
            and shows the fallback in use.
            """
        )

        st.header("Technology Stack")
        st.markdown(
            """
            | Component | Technology | Version |
            |-----------|-----------|---------|
            | Model | CatBoost | ≥ 1.2 |
            | Feature selection | PCMCI (tigramite) | Research phase only |
            | Data ingestion | requests + pandas | — |
            | Dashboard | Streamlit + Plotly | ≥ 1.32 |
            | Linting | Ruff | ≥ 0.4 |
            | Type checking | mypy | ≥ 1.9 |
            | Pre-commit hooks | ruff, trailing-whitespace, detect-private-key | — |
            | Weather API | OpenWeatherMap (free tier) | — |
            | Load data | ISO-NE public CSV reports | — |
            """
        )


# ---------------------------------------------------------------------------
# Pipeline diagram (styled HTML)
# ---------------------------------------------------------------------------

_PIPELINE_HTML = """\
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:12px 0">

  <style>
  .pbox{border-radius:10px;padding:12px 16px;text-align:center;min-width:140px;
    box-shadow:0 2px 6px rgba(0,0,0,.07);line-height:1.4;display:inline-block;vertical-align:top}
  .pbox b{display:block;font-size:.9rem;margin-bottom:2px}
  .pbox span{font-size:.75rem;opacity:.8}
  .parr{font-size:1.4rem;color:#9aa0a6;vertical-align:middle;margin:0 6px}
  .parr-d{font-size:1.3rem;color:#9aa0a6;display:block;text-align:center;margin:4px 0}
  .prow{text-align:center;white-space:nowrap;margin-bottom:4px}
  .psrc{background:linear-gradient(135deg,#e8f0fe,#d2e3fc);border:1.5px solid #4285f4;color:#1a3563}
  .pprc{background:linear-gradient(135deg,#e6f4ea,#ceead6);border:1.5px solid #34a853;color:#1a4731}
  .pfeat{background:linear-gradient(135deg,#fef7e0,#feefc3);border:1.5px solid #f9ab00;color:#594300}
  .pmod{background:linear-gradient(135deg,#fce8e6,#f8d0cb);border:1.5px solid #ea4335;color:#5f1612}
  .pout{background:linear-gradient(135deg,#f3e8fd,#e8d5f5);border:1.5px solid #9334e6;color:#3b1272}
  </style>

  <!-- Row 1 -->
  <div class="prow">
    <div style="display:inline-flex;flex-direction:column;gap:6px;vertical-align:middle">
      <div class="pbox psrc"><b>ISO-NE Data</b><span>Hourly RTLO load<br>public CSV reports</span></div>
      <div class="pbox psrc"><b>OpenWeatherMap</b><span>Temp, humidity<br>wind, pressure</span></div>
    </div>
    <span class="parr">&rarr;</span>
    <div class="pbox pprc" style="vertical-align:middle"><b>Merge &amp; Clean</b><span>Inner join on datetime<br>Drop high-missing cols<br>Median imputation</span></div>
    <span class="parr">&rarr;</span>
    <div class="pbox pfeat" style="vertical-align:middle"><b>Feature Engineering</b><span>Calendar (cyclic h/dow/m)<br>Holidays (US + MA)<br>HDD / CDD / sigmoids</span></div>
  </div>

  <div class="parr-d">&darr;</div>

  <!-- Row 2 -->
  <div class="prow">
    <div class="pbox pmod"><b>168 h Lookback</b><span>Lag extraction<br>lag-4 &middot; mean-24<br>lag-168 &middot; mean-168</span></div>
    <span class="parr">&rarr;</span>
    <div class="pbox pmod"><b>CatBoost Regressor</b><span>1 000 trees &middot; depth 8<br>lr 0.05 &middot; MAE loss<br>early stop @ 50</span></div>
    <span class="parr">&rarr;</span>
    <div class="pbox pout"><b>24 h Forecast</b><span>Compared hour-by-hour<br>vs ISO-NE official<br>3-day forecast</span></div>
    <span class="parr">&rarr;</span>
    <div class="pbox pout"><b>Dashboard</b><span>Live metrics &amp; charts<br>rolling MAE, diagnostics<br>model documentation</span></div>
  </div>

</div>
"""
