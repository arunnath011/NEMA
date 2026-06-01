# NEMA Load Forecasting Pipeline

Production-grade 24-hour-ahead electricity load forecasting for the **NEMA (Northeast Massachusetts and Boston)** zone in ISO New England, powered by CatBoost gradient boosting.

## Performance

| Model | MAE (MW) | MAPE (%) | R² |
|-------|----------|----------|-----|
| **CatBoost** | ~106 | ~3.85 | 0.93 |
| ISO-NE Forecast | ~180 | ~6.5 | 0.82 |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd nema-forecast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure API keys
cp .env.example .env
# Edit .env with your OpenWeatherMap API key and ISO-NE Web Services credentials

# 3. Backfill the historical demand series (one-time; uses your ISO-NE login)
#    Start with a short range to validate, then widen to the full history.
python -m nema_forecast.scripts.backfill_load --start 2023-01-01
python -m nema_forecast.scripts.backfill_load --start 2017-03-01 --with-dayahead

# 4. Train the model on the backfilled series
python -m nema_forecast.model.train

# 5. Run the backtest (CatBoost vs ISO-NE day-ahead demand)
python -m nema_forecast.model.backtest

# 6. Launch the dashboard
streamlit run src/nema_forecast/dashboard/app.py
```

## Data Sources

| Source | URL | Access |
|--------|-----|--------|
| ISO-NE Real-Time Demand | [Web Services API](https://www.iso-ne.com/participate/support/web-services-data) | Free registration, HTTP Basic auth |
| ISO-NE Day-Ahead Demand | [Web Services API](https://www.iso-ne.com/participate/support/web-services-data) | Free registration, HTTP Basic auth |
| EIA Open Data (fallback) | [eia.gov/opendata](https://www.eia.gov/opendata/register.php) | Free API key (instant) |
| OpenWeatherMap | [openweathermap.org/api](https://openweathermap.org/api) | Free API key |

**Live load** comes from the ISO-NE Web Services API (`realtimehourlydemand`, location 4008 =
NEMASSBOST), which publishes near-real-time hourly demand with ~1 hour latency — not the legacy
WHLSECOST settlement report (delayed 4–6 weeks). The model trains and serves on this same series.
The `dayaheadhourlydemand` feed is used as the ISO-NE benchmark the model is scored against.

**Fallback:** if ISO-NE Web Services credentials aren't set (registration needs approval), the
dashboard automatically falls back to the **EIA Open Data API** (`ISNE`/`4008` subregion = true
NEMA-zone hourly demand, ~1–2 h latency), whose free key is issued instantly. The live page shows
which source supplied the data. Source selection lives in
[`data/load_source.py`](src/nema_forecast/data/load_source.py).

## Project Structure

```
nema-forecast/
├── src/nema_forecast/
│   ├── config.py                    # Paths, API endpoints, constants
│   ├── data/
│   │   ├── load_source.py           # Live source facade (ISO-NE WS → EIA fallback)
│   │   ├── iso_ne_ws.py             # ISO-NE Web Services client (primary live source)
│   │   ├── eia.py                   # EIA Open Data client (fallback live source)
│   │   ├── iso_ne.py                # Legacy WHLSECOST CSV fetcher (deprecated)
│   │   ├── weather.py               # OpenWeatherMap client
│   │   └── preprocessing.py         # Cleaning, imputation
│   ├── scripts/
│   │   └── backfill_load.py         # One-time historical demand backfill
│   ├── features/
│   │   └── engineering.py           # Calendar, weather, lag features
│   ├── model/
│   │   ├── train.py                 # Training pipeline
│   │   ├── inference.py             # 24h prediction
│   │   └── backtest.py              # Rolling evaluation
│   └── dashboard/
│       ├── app.py                   # Streamlit entry point
│       ├── components.py            # Reusable UI components
│       └── pages/                   # 5 dashboard pages
├── models/                          # Serialised model artefacts
├── data/cache/                      # Cached API responses
├── tests/                           # Pytest suite
├── pyproject.toml                   # Build config + tool settings
└── .pre-commit-config.yaml          # Ruff, mypy, pre-commit hooks
```

## Dashboard Pages

1. **Executive Summary** — KPI cards, rolling performance, latest predictions
2. **Live Forecast** — Current weather + real-time 24h ahead forecast
3. **Model vs ISO-NE** — Side-by-side comparison with scatter, error, hourly breakdowns
4. **Diagnostics** — Feature importance, residuals, autocorrelation, leakage tests
5. **How The Model Works** — Comprehensive technical documentation

## Deploy to Streamlit Cloud

The dashboard is ready for one-click deployment to [Streamlit Community Cloud](https://share.streamlit.io) (free).

1. **Push to GitHub** — make sure your repo includes the `models/` artefacts
2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. **New app** → select your repo, branch `main`, main file path:
   ```
   src/nema_forecast/dashboard/app.py
   ```
4. **Add secrets** — in *Advanced settings → Secrets*, paste:
   ```toml
   OWM_API_KEY = "your_openweathermap_api_key"
   ISO_NE_WS_USER = "your_iso_ne_username"   # primary live source
   ISO_NE_WS_PASS = "your_iso_ne_password"
   EIA_API_KEY = "your_eia_api_key"          # instant-key fallback source
   ```
5. **Deploy** — the app will be live at `https://<your-name>-nema-forecast.streamlit.app`

The dashboard auto-updates on every visit: ISO-NE real-time demand is fetched live from the Web
Services API (with a 1-hour cache), weather comes from OpenWeatherMap in real time, and the
recursive 24-hour CatBoost forecast runs server-side. No local data files are needed at runtime —
only the trained `models/` artefacts, which must be committed to the repo.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run linting
ruff check src/ tests/
ruff format src/ tests/

# Run tests
pytest

# Type checking
mypy src/
```

## License

MIT
