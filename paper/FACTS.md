# Ground-Truth Facts & Results — Beacon NEMA Load Forecasting Paper

**This file is the single source of truth for the paper. Writing agents MUST use only the
numbers and claims here. Do not invent statistics. If a number is not here, do not state it.**

All results are reproducible from the repository (`arunnath011/NEMA`).

---

## 1. Problem & contribution

- **Task:** 24-hour-ahead hourly electricity load (demand) forecasting for the **NEMA**
  zone (Northeast Massachusetts / Boston) of **ISO New England** — load zone `.Z.NEMASSBOST`,
  ISO-NE location id **4008**.
- **Core contributions:**
  1. A **direct multi-horizon** gradient-boosting forecaster ("Beacon"): 24 CatBoost models,
     one per horizon h=1…24, each predicting its step directly from a 168-hour lookback window
     (no recursive roll-out, so no error compounding).
  2. **Target-hour exogenous features**: each horizon model receives the calendar features
     (exactly known) and the *forecasted weather* at the target hour t+h — the dominant driver
     of load a day ahead.
  3. **Train/serve weather-source unification** on **Open-Meteo** (free, keyless): the same
     source for training history (ERA5 archive) and live serving (forecast), eliminating a
     train/serve distribution mismatch that otherwise nearly doubled day-ahead error.
  4. **Horizon-matched, honest evaluation** against ISO-NE's operational day-ahead forecast,
     correcting a common pitfall (comparing a model's 1-hour nowcast against a day-ahead
     benchmark).

## 2. Data

- **Load (training):** ISO-NE hourly wholesale-load (RTLO) for NEMA, **2017-03-01 → 2025-11-30**,
  **76,719 hourly observations** after cleaning.
- **Load (live serving / deployment):** ISO-NE Web Services API `realtimehourlydemand`
  (location 4008); publishes with a ~4-day settlement lag; EIA hourly demand as fallback.
- **Weather:** Open-Meteo, Boston Logan (42.3656° N, −71.0096° W). Variables: temperature,
  relative humidity, wind speed, dew point, cloud cover, apparent temperature, visibility
  (°F / mph / %). Archive API (ERA5) for history; forecast API for serving. No API key.
- **Benchmark:** ISO-NE Web Services `dayaheadhourlydemand` (location 4008) — ISO-NE's
  official day-ahead demand forecast for NEMA.
- **Train/test split:** strict temporal split at **TRAIN_CUTOFF = 2024-12-31 23:00**.
  Train = 68,704 rows; Test = 8,015 rows. After sequence creation: **58,236 train / 10,276
  validation / 7,823 test** windows. (Validation = last 15% of training.)

## 3. Features

- **Lookback window:** 168 hours (1 week).
- **Window lag features (173 total):** for each engineered channel, values at lags
  {1,4,8,24,48,168} h and rolling means over {24,168} h, with an RTLO whitelist kept from the
  notebook ablation: `RTLO_lag4, RTLO_mean24, RTLO_lag48, RTLO_lag168, RTLO_mean168`.
- **Calendar:** cyclic encodings of hour, day-of-week, month (sin/cos), `is_weekend`,
  `is_us_holiday`.
- **Weather-derived:** Heating/Cooling Degree Days `HDD=max(0,65−T)`, `CDD=max(0,T−65)`
  (base 65 °F), `temp²`, `HDD²`, `CDD²`, and sigmoidal thermal-stress transforms (logistic,
  centred at 18 °C heating / 22 °C cooling).
- **Target-hour exogenous features (21):** all engineered calendar + weather features at the
  **target hour t+h** (forecast weather + exactly-known calendar), **excluding RTLO** (RTLO at
  t+h is the target). **Augmented feature vector = 173 + 21 = 194 features.**

## 4. Model & training

- **Learner:** CatBoost gradient-boosted trees. Hyper-parameters: 1000 iterations, depth 8,
  learning rate 0.05, **MAE loss**, early stopping after 50 rounds, seed 42.
- **Strategy:** **direct multi-horizon** — 24 independent models, model_h predicts load at
  t+h from [window lag features ⊕ target-hour exogenous features at t+h].
- **Baseline (for ablation):** a single h=1 model whose prediction is reused for every horizon
  ("single model rolled out").

## 5. Headline results — held-out test set (7,823 windows, 2025)

| Metric | Value |
|---|---|
| MAE @ h=1 | **60.9 MW** |
| MAE @ h=24 (day-ahead) | **76.7 MW** |
| Average MAE across 24 h | **73.6 MW** |
| MAPE (h=1) | **2.25 %** |
| R² (h=1) | **0.977** |
| Horizon degradation (MAE h24 / MAE h1) | **1.26×** |
| Avg improvement vs single-model baseline (across 24 h) | **80.6 %** |

## 6. Per-horizon MAE (test set) — Beacon vs single-model baseline

| Horizon h | Beacon direct MAE (MW) | Single-model rolled-out MAE (MW) |
|---|---|---|
| 1  | 60.9 | 62.6 |
| 4  | 71.3 | 234.8 |
| 6  | 74.1 | 351.7 |
| 12 | 76.7 | 505.0 |
| 18 | 76.3 | 480.8 |
| 24 | 76.7 | 234.8 |

The single-model baseline degrades catastrophically at mid-horizons (load at t+1 is a poor
proxy for the opposite phase of the diurnal cycle), peaking ~505 MW near h=12; Beacon stays
flat ~60–77 MW.

## 7. Live, horizon-matched evaluation vs ISO-NE (30-day window, day-ahead)

Both forecasters evaluated at the **same 24-hour-ahead horizon**, using the same Open-Meteo
weather, over the most recent ~30 days (720 hourly points). Actuals = ISO-NE real-time demand.

| Forecaster | MAE (MW) | MAPE (%) | R² |
|---|---|---|---|
| **Beacon (day-ahead)** | **89.3** | **3.50** | 0.904 |
| ISO-NE (day-ahead) | 92.6 | 3.70 | 0.923 |

**Beacon beats ISO-NE on MAE (−3.3 MW, ≈ +3.5%) and MAPE**; ISO-NE has marginally higher R².
This is the honest, like-for-like result.

## 8. Ablations (test-set MAE, MW)

### 8.1 Effect of target-hour features and weather source (cumulative)

| Configuration | MAE @ h=1 | MAE @ h=24 |
|---|---|---|
| Direct multi-horizon, lags only, OWM weather | 78.7 | 183.0 |
| + target-hour features (calendar + weather), OWM weather | 77.7 | 110.3 |
| + Open-Meteo weather (train/serve matched) — **final** | 60.9 | 76.7 |

Target-hour weather features cut day-ahead MAE 183 → 110 (≈ −40%); unifying the weather source
on Open-Meteo cut it further 110 → 77, and improved h=1 (77.7 → 60.9).

### 8.2 Weather-source mismatch (live day-ahead)

Trained on OpenWeatherMap CSVs but served Open-Meteo weather → live day-ahead MAE **≈ 147 MW**.
Retrained on Open-Meteo (matched to serving) → live day-ahead MAE **≈ 90 MW** (vs ISO ≈ 92).
The single change of *matching the train/serve weather source* removed most of the gap to ISO.

### 8.3 Forecast strategy

Direct per-horizon vs single-model rolled-out: **80.6%** lower average MAE across the 24-hour
horizon (Section 6).

## 9. Data-leakage validation

- **Structural:** every lag/rolling feature references data at or before t−1; the target for
  horizon h is at t+h ≥ t. No feature can contain target-time information. Train/test are
  temporally separated at the cutoff.
- **Shuffled-target test** (gold standard): models trained on randomly permuted targets.

| Horizon | Real-target test R² | Shuffled-target test R² |
|---|---|---|
| h=1 | 0.956 | −0.019 |
| h=24 | 0.770 | −0.021 |

R² collapses to ≈ 0 under shuffling at both horizons → the model exploits genuine
feature–target structure, not leakage. The monotone increase of MAE with horizon (Section 6)
is itself further evidence against leakage (a leaking model would be near-flat).

## 10. Deployment / system

- **Live dashboard:** Streamlit (5 pages: executive summary, live forecast, model-vs-ISO,
  diagnostics, methodology). Hosted-ready (Streamlit Community Cloud).
- **Live data:** ISO-NE Web Services (`realtimehourlydemand`, `dayaheadhourlydemand`) with
  HTTP-Basic auth; EIA Open Data API fallback (ISNE/4008); 429 rate-limit retry with
  exponential backoff. Weather from Open-Meteo (keyless).
- **Reproducibility:** code at `github.com/arunnath011/NEMA`; training
  (`python -m nema_forecast.model.train`) produces the 24 per-horizon models and all metric
  artefacts (`model_performance.json`, `horizon_mae.json`).

## 11. Framing / positioning (for Introduction & Discussion)

- Load forecasting is critical for unit commitment, economic dispatch, and reserve sizing in
  wholesale electricity markets; small MAE reductions translate to large cost/reliability gains.
- ISO-NE publishes an operational day-ahead reliability-region demand forecast that operators
  rely on; beating it is a meaningful, hard bar.
- Two methodological pitfalls this work highlights and fixes: (i) **horizon mismatch** in
  model-vs-operator comparisons; (ii) **train/serve weather-source mismatch** that silently
  degrades deployed accuracy.
- Limitations: single zone (NEMA) and a relatively short live evaluation window (30 days,
  spring/early-summer); WHLSECOST vs real-time-demand series differ slightly; Open-Meteo
  archive omits visibility (filled with a constant); point forecasts only (no probabilistic
  intervals yet).
- Future work: probabilistic/quantile bands, ensembling with a transformer (Informer),
  multi-zone extension, Optuna hyper-parameter search, recency-weighted retraining.
