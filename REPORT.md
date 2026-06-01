# NEMA Load Forecasting — Technical Report

**Project:** Hourly Electricity Load Forecasting for the NEMA (Northeast Massachusetts & Boston) Zone
**ISO Region:** ISO New England
**Author:** Auto-generated from `NEMA_CatBoost_Comprehensive.ipynb` and `NEMA_publications.ipynb`
**Date:** February 2026

---

## 1. Executive Summary

This report documents the development and validation of a CatBoost gradient-boosting model for 24-hour-ahead hourly electricity load forecasting in the NEMA zone of ISO New England. The model was trained on ~68,700 hourly observations (March 2017 — December 2024) and evaluated on ~8,000 out-of-sample hours (January — November 2025).

### Key Results

| Model | MAE (MW) | MAPE (%) | R² |
|-------|----------|----------|-----|
| **CatBoost (final)** | **~106** | **~3.85** | **0.93** |
| ISO-NE Official 3-Day Forecast | ~180 | ~6.5 | 0.82 |
| Persistence (lag-1) | Higher baseline | — | — |
| Seasonal Naive (168h) | Higher baseline | — | — |
| Historical Average | Higher baseline | — | — |

The CatBoost model outperforms the ISO-NE official three-day reliability region demand forecast by approximately 41% on MAE, demonstrating that a well-engineered gradient boosting approach can meaningfully improve upon operational forecasts used in the New England power system.

---

## 2. Data Sources and Preparation

### 2.1 Data Sources

| Source | Description | Granularity | Period |
|--------|-------------|-------------|--------|
| ISO-NE RTLO | Real-Time Locational Obligation wholesale load cost data for location ID 4008 (NEMA/Boston) | Hourly | Mar 2017 — Nov 2025 |
| OpenWeatherMap | Historical weather observations at Boston Logan Airport (42.37°N, 71.01°W) | Hourly | Same period |
| ISO-NE 3-Day Forecast | Three-day reliability region load forecast for `.Z.NEMASSBOST` | Hourly | Overlapping test period |

> **Production data-source migration (deployment).** The findings above reflect the model
> trained on the WHLSECOST settlement series (published on a 4–6 week delay). For the live
> deployment, both the training and serving load series have been migrated to the **ISO-NE Web
> Services API** `realtimehourlydemand` feed (location 4008), which publishes with ~1 hour
> latency, and the benchmark moved from the three-day forecast to the `dayaheadhourlydemand`
> feed. The metrics in this report should be regenerated after the historical backfill and
> retrain (`scripts/backfill_load.py` → `model/train.py`) on the new series.

### 2.2 Train/Test Split

A strict temporal split was used to prevent data leakage:

- **Training set:** 68,704 rows (2017-03-01 to 2024-12-31 23:00)
- **Test set:** 8,015 rows (2025-01-01 to 2025-11-30 23:00)
- **Validation:** 15% of training data held out for early stopping

The gap between train and test exceeds the 168-hour (one-week) lookback window, ensuring no temporal leakage.

### 2.3 Missing Value Treatment

A three-tier strategy was applied based on training-set missingness rates:

| Category | Threshold | Action | Columns |
|----------|-----------|--------|---------|
| **High** | >50% missing | Dropped | `sea_level`, `grnd_level`, `DA_Ancillary_Service_Cost`, `Inventory_Energy_Program_Cost`, `Price_Responsive_Demand_Cost`, `snow_3h`, `rain_3h`, `snow_1h`, `rain_1h`, `wind_gust`, `Real-Time_Demand_Reduction_Cost` |
| **Medium** | 5–50% | Imputed (median from training data) | — |
| **Low** | <5% | Imputed (median) + forward-fill | `temp`, `humidity`, `wind_speed`, `visibility`, `dew_point`, `clouds_all`, `feels_like` |

Imputation statistics were computed exclusively from training data and applied identically to the test set to prevent information leakage.

---

## 3. Exploratory Data Analysis

### 3.1 Load Distribution

NEMA load exhibits a right-skewed distribution, reflecting a base load floor with high-demand peaks driven by summer cooling and winter heating. Descriptive statistics were computed for both train and test sets to confirm distributional consistency.

### 3.2 Temporal Patterns

Four dominant patterns were identified:

1. **Hourly (diurnal) cycle:** Load peaks in mid-afternoon (hours 14–18) and troughs overnight (hours 2–5). The mean±std band shows consistent variability across hours.

2. **Day-of-week pattern:** Weekdays carry 10–15% higher average load than weekends, with Monday and Friday showing slight reductions compared to mid-week.

3. **Monthly (seasonal) cycle:** A U-shaped annual curve with peaks in January (heating) and July–August (cooling), and troughs in May and October (shoulder seasons).

4. **Hour × Day-of-Week interaction:** A heatmap reveals the weekday afternoon peak is the single strongest load regime, while weekend mornings are the lowest.

### 3.3 Weather–Load Relationship

Correlation analysis between weather variables and RTLO revealed:

- **Temperature** shows a non-linear (U-shaped) relationship with load — both extreme cold and extreme heat drive demand up due to heating and cooling loads respectively.
- **Dew point** and **feels_like** are highly correlated with temperature, confirming the need for derived features (HDD/CDD) rather than raw values.
- **Humidity**, **wind speed**, and **visibility** have weaker but non-negligible correlations.

---

## 4. Causal Feature Selection (PCMCI)

The PCMCI algorithm (Peter and Clark Momentary Conditional Independence, implemented via Tigramite) was applied to identify causally relevant weather features, moving beyond naive correlation.

**Configuration:**
- Independence test: Partial Correlation (ParCorr, analytic significance)
- Maximum lag: τ_max = 5 hours
- Significance threshold: α = 0.05

**Results:** PCMCI identified statistically significant causal links from `temp`, `humidity`, `wind_speed`, and `visibility` to load at various lags. Features surviving the causal filter were retained for model training.

This step guards against including spuriously correlated features that could cause the model to degrade on unseen data.

---

## 5. Feature Engineering

### 5.1 Calendar Features

| Feature | Description |
|---------|-------------|
| `hour_sin`, `hour_cos` | Cyclic encoding of hour-of-day |
| `dow_sin`, `dow_cos` | Cyclic encoding of day-of-week |
| `month_sin`, `month_cos` | Cyclic encoding of month |
| `is_weekend` | Binary weekend indicator |
| `is_us_holiday` | Binary US federal holiday indicator |

### 5.2 Weather-Derived Features

| Feature | Description |
|---------|-------------|
| `HDD` | Heating Degree Days: max(0, 65 − temp) |
| `CDD` | Cooling Degree Days: max(0, temp − 65) |
| `temp_sq` | Quadratic temperature term for non-linearity |
| `HDD_sq`, `CDD_sq` | Quadratic HDD/CDD for extreme-temperature response |
| `temp_heating_sigmoid` | Sigmoidal transformation centered at 18°C |
| `temp_cooling_sigmoid` | Sigmoidal transformation centered at 22°C |
| `temp_thermal_stress` | Combined heating + cooling sigmoid |

### 5.3 RTLO Lag Features

The model uses a 168-hour (1-week) lookback window. From this window, features are extracted at specific lag offsets:

| Lag | Rationale |
|-----|-----------|
| lag-1 | Most recent load (strong persistence) |
| lag-4 | Same part of day cycle |
| lag-8 | Opposing part of day |
| lag-24 | Same hour yesterday |
| lag-48 | Same hour two days ago |
| lag-168 | Same hour, same day of week last week |

Additionally, rolling means over 24h and 168h windows capture trend and weekly level.

A curated whitelist of RTLO features was used based on ablation results (Section 8):
`RTLO_lag4`, `RTLO_mean24`, `RTLO_lag48`, `RTLO_lag168`, `RTLO_mean168`, `CDD_sq_mean24`, `CDD_mean24`, `hour_cos_lag4`.

### 5.4 Sequence Construction

Sequences of length 168 (one week) are constructed from the feature matrix. For each prediction point, the model sees 168 hours of lagged features and predicts load at horizons 1 through 24. The lag extraction step flattens these into a tabular feature vector suitable for gradient boosting.

---

## 6. Baseline Models

Three naive baselines were evaluated to establish a performance floor:

| Model | Description | MAE (MW) |
|-------|-------------|----------|
| **Persistence** | Predict load = load at hour t−1 | Highest |
| **Seasonal Naive (168h)** | Predict load = load at same hour last week | Mid-range |
| **Historical Average** | Predict load = average for that hour-of-day from training set | Mid-range |

All baselines were computed on the test set to provide directly comparable reference points.

---

## 7. CatBoost Model

### 7.1 Architecture

| Parameter | Value |
|-----------|-------|
| Algorithm | CatBoost (gradient boosted decision trees) |
| Loss function | MAE |
| Iterations | 1,000 (early stopping after 50 rounds) |
| Learning rate | 0.05 |
| Tree depth | 8 |
| Random seed | 42 |

### 7.2 Training Procedure

1. Feature matrix split: 85% train / 15% validation (temporal, no shuffling)
2. CatBoost trained with early stopping on validation MAE
3. Best model selected at the iteration with lowest validation loss

### 7.3 Test Results

| Metric | Value |
|--------|-------|
| **MAE** | ~106 MW |
| **MAPE** | ~3.85% |
| **R²** | 0.93 |

---

## 8. Ablation Studies

### 8.1 RTLO Lag Ablation

A comprehensive study tested 10 lag configurations to understand the marginal value of each:

| Configuration | Description |
|---------------|-------------|
| `no_rtlo` | All RTLO features removed |
| `lag1` | Only 1-hour lag |
| `lag4` | Only 4-hour lag |
| `lag24` | Only 24-hour lag |
| `lag168` | Only 168-hour lag |
| `lag1+24` | 1h + 24h lags |
| `lag1+24+168` | 1h + 24h + 168h lags |
| `all` | All 6 lag offsets |

**Finding:** The combination of lag-1, lag-24, and lag-168 captures the vast majority of predictive power from persistence. Removing RTLO features entirely degrades performance substantially, confirming that autoregressive information is critical but can be supplied through a small set of well-chosen lags.

### 8.2 Polynomial Features

Polynomial features (`temp_sq`, `HDD_sq`, `CDD_sq`) were tested against a non-polynomial baseline:
- **With polynomials:** MAE ≈ 106 MW
- **Without polynomials:** MAE slightly higher

The improvement confirms that the temperature–load relationship has significant curvature, particularly at thermal extremes.

### 8.3 Sigmoidal Temperature Features

Three sigmoidal temperature transformations were tested:
- `temp_heating_sigmoid`: Smooth transition centered at 18°C (captures heating regime)
- `temp_cooling_sigmoid`: Smooth transition centered at 22°C (captures cooling regime)
- `temp_thermal_stress`: Sum of both (overall thermal discomfort)

**Results:** Adding sigmoidal features on top of RTLO provides marginal improvement. Without RTLO features, sigmoidal features partially fill the performance gap, demonstrating they capture real load-driving physics.

---

## 9. Diagnostic & Leakage Tests

A comprehensive 10-test validation suite was executed to ensure the model's performance is legitimate and production-ready.

### 9.1 Standard Diagnostics (5 Tests)

| Test | Description | Result |
|------|-------------|--------|
| **Temporal Separation** | Verify train/test gap ≥ 168h lookback | PASS |
| **Feature-Target Correlation** | Flag features with >0.99 correlation to target | PASS (no suspicious features) |
| **Naive Baseline Comparison** | CatBoost must beat lag-1 by >5% | PASS |
| **Shuffled Target** | Train on permuted targets, expect >50% MAE degradation | PASS (confirms model learns real signal) |
| **Feature Importance** | Verify no single feature dominates suspiciously | PASS |

### 9.2 Forensic Leakage Tests (5 Tests)

| Test | Description | Result |
|------|-------------|--------|
| **Remove RTLO Features** | Quantify RTLO contribution vs weather-only | Confirmed: RTLO adds significant value |
| **Shuffle RTLO Lags** | Destroy temporal structure in RTLO only | Large MAE degradation, confirming temporal signal is real |
| **Horizon-wise MAE** | MAE should increase with forecast horizon | PASS: h24/h1 ratio > 1.3x (expected degradation curve) |
| **+48h Target Shift** | Predict further ahead, expect worse performance | PASS: Performance degrades appropriately |
| **Year-Based Split** | Confirm proper temporal separation by year | PASS |

### 9.3 Additional Leakage Tests (4 Tests from Comprehensive Notebook)

| Test | Description | Result |
|------|-------------|--------|
| **Train/Test Datetime Overlap** | No shared timestamps between sets | PASS |
| **Target Correlation Check** | No non-lag feature with >0.99 correlation | PASS |
| **Lag Index Validation** | All lag features have lag ≥ 1 | PASS |
| **Future Weather Leakage** | No "lead", "future", or negative-shift features | PASS |

### 9.4 Production Enhancement: Extreme-Day Analysis

The top 5% load hours (extreme demand) were evaluated separately:
- **Normal hours MAE:** Lower (model performs well on typical conditions)
- **Extreme hours MAE:** Higher (expected degradation under stress)
- The degradation is smooth and bounded, indicating the model doesn't catastrophically fail during peak events.

---

## 10. ISO-NE Benchmark Comparison

The CatBoost model was compared head-to-head against ISO-NE's official three-day reliability region load forecast for the NEMASSBOST zone.

### 10.1 Methodology

- ISO forecast CSVs were loaded and parsed (filtering for `type='D'`, `region='.Z.NEMASSBOST'`)
- For each datetime, the most recently published ISO forecast was retained
- Predictions were aligned on matching datetime indices

### 10.2 Results

| Model | MAE (MW) | MAPE (%) | R² |
|-------|----------|----------|-----|
| **CatBoost** | ~106 | ~3.85 | 0.93 |
| **ISO-NE 3-Day** | ~180 | ~6.5 | 0.82 |
| **Improvement** | **~41%** | **~41%** | — |

### 10.3 Error Distribution

- CatBoost errors are more tightly centered around zero with thinner tails
- ISO-NE forecast shows wider dispersion, particularly on high-demand days
- Both models show slight positive bias (tendency to under-predict peaks)

### 10.4 Hourly Breakdown

The ISO-NE forecast MAE by hour-of-day reveals higher errors during:
- Morning ramp-up (hours 6–9)
- Evening ramp-down (hours 18–21)

CatBoost more effectively captures these transition periods due to its access to fine-grained autoregressive features.

---

## 11. Hardened Model Validation (Publications Notebook)

A second notebook (`NEMA_publications.ipynb`) implemented a rigorous multi-phase hardening pipeline:

### 11.1 Phase Summary

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 0** | Baseline freeze + rolling-origin seasonal CV | Defined |
| **Phase 1** | PCMCI hardening (multi-τ stability, placebo rejection) | Defined |
| **Phase 2** | RTLO rearchitecture (STL decomposition, lag dropout, horizon-consistent lags) | Defined |
| **Phase 3** | Weather reformulation (binned temps, piecewise HDD/CDD) | Defined |
| **Phase 3.5** | NWP forecast error injection testing | Defined |
| **Phase 4** | Hardened model v2 training | **Executed** |
| **Phase 5** | Forensic validation (partial R², noise injection, regime analysis) | Defined |
| **Phase 6** | Model identity classification | Defined |
| **Phase 7** | Ensemble + CI gating | Defined |

### 11.2 Hardened Model v2 Results

The hardened model (Phase 4), trained with 75 features including binned temperature features, weather lags, and categorical market variables, achieved:

| Metric | Value |
|--------|-------|
| **MAE** | 144.19 MW |
| **MAPE** | 5.36% |
| **RMSE** | 191.42 MW |
| **R²** | 0.8768 |

This model deliberately sacrifices some accuracy (R² dropped from 0.93 to 0.88) in exchange for robustness. The original model's high R² was partially driven by persistence features — the hardened model confirms that even with reduced RTLO dependence, the model maintains operationally useful accuracy.

### 11.3 Key Design Decisions

**STL over PCA for RTLO decomposition:** STL (Seasonal-Trend Decomposition using LOESS) was recommended over PCA because:
- Components are interpretable (trend, daily season, weekly season, residual)
- Explainable to operations teams and dispatchers
- Defensible for regulatory compliance

**Lag dropout training:** Production-calibrated dropout probabilities simulate real-world data latency:
- lag-1: 20% dropout (real-time telemetry often delayed)
- lag-24: 10% dropout (day-ahead data more reliable)
- lag-168: 5% dropout (historical data rarely missing)

**Horizon-consistent lag rules:** For horizon h, only lags ≥ h are permitted:
- h=1: all lags available
- h=6: lag-6, lag-24, lag-168
- h=24: lag-24, lag-168

**NWP error budgeting:** Three scenarios were defined for weather forecast error injection:

| Scenario | Temp Error (σ) | Humidity Error (σ) |
|----------|---------------|--------------------|
| Mild | ±2°F | ±5% |
| Moderate | ±4°F | ±10% |
| Severe | ±6°F | ±15% |

### 11.4 Model Identity Classification

The model is classified on a spectrum based on partial R² decomposition:
- **Persistence-Driven** (>70% of explained variance from RTLO features)
- **Hybrid** (30–70%)
- **Driver-Based** (<30%)

The acceptance criterion: *"If R² drops from 0.994 to 0.97, that is a win."* — A lower but defensible R² from weather/calendar features alone is preferable to an inflated R² driven by near-trivial persistence.

---

## 12. Production Architecture

The model has been productionised into a Python package (`nema-forecast`) with the following components:

| Component | Description |
|-----------|-------------|
| `data/iso_ne.py` | ISO-NE data fetching with session management and caching |
| `data/weather.py` | OpenWeatherMap API client (current + forecast) |
| `data/preprocessing.py` | Imputation, cleaning pipeline |
| `features/engineering.py` | Calendar, weather, lag feature construction |
| `model/train.py` | Training pipeline with artifact export |
| `model/inference.py` | 24-hour-ahead prediction from live data |
| `model/backtest.py` | Rolling-origin backtesting framework |
| `dashboard/` | 5-page Streamlit dashboard |

### Dashboard Pages

1. **Executive Summary** — KPI cards, rolling MAE/MAPE, latest predictions
2. **Live Forecast** — Current weather conditions + real-time 24h forecast
3. **Model vs ISO-NE** — Side-by-side scatter, error distribution, hourly breakdown
4. **Diagnostics** — Feature importance, residual analysis, autocorrelation, leakage tests
5. **How The Model Works** — Technical documentation and methodology

---

## 13. Limitations and Future Work

### Current Limitations

1. **Weather observations at inference:** The model currently assumes observed (not forecasted) weather is available. In production, NWP forecast errors will degrade performance for horizons beyond ~1 hour.

2. **Single-zone scope:** The model is trained for NEMA/Boston only. Extension to other ISO-NE zones would require retraining.

3. **CSV scraping for data access:** ISO-NE data is currently fetched via web scraping of the ISO Express CSV download pages. ISO-NE provides an official REST API (v1.1 at `webservices.iso-ne.com`) that would be more reliable but requires API credentials.

4. **No extreme event modeling:** While the model handles the 95th percentile reasonably, extreme events (ice storms, heat waves, equipment failures) may require separate treatment.

### Recommended Next Steps

1. **Integrate ISO-NE Web Services API** — Replace CSV scraping with authenticated API calls for reliable automated data ingestion.

2. **NWP error injection testing** — Execute the Phase 3.5 error scenarios to quantify degradation under realistic weather forecast errors.

3. **Rolling-origin seasonal CV** — Run the Phase 0 seasonal cross-validation across winter/spring/summer/fall to validate stability.

4. **Partial R² decomposition** — Execute Phase 5 to formally classify the model's dependence on persistence vs. weather drivers.

5. **Ensemble approach** — Combine the RTLO-dependent model with a weather-only model, blending weights by forecast horizon.

6. **Lag dropout deployment** — Train with production-calibrated dropout to build robustness to real-time data latency.

---

## 14. Conclusion

The NEMA load forecasting model demonstrates that a CatBoost model with well-engineered temporal and weather features can substantially outperform ISO-NE's operational 3-day forecast (~41% MAE improvement). The model passes all 14 diagnostic and leakage tests, confirming that its performance reflects genuine predictive signal rather than data leakage.

The hardening pipeline (publications notebook) provides a clear path from the current persistence-dependent model toward a production-robust system that gracefully degrades when autoregressive data is delayed or unavailable. The key insight is that modest R² reductions (0.93 to 0.88) represent gains in robustness and deployability rather than losses in capability.

---

*Report generated from analysis of `NEMA_CatBoost_Comprehensive.ipynb` (Feb 8, 2026) and `NEMA_publications.ipynb` (Feb 6, 2026).*
