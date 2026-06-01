"""Rolling backtest over the test period — compares CatBoost vs ISO-NE forecasts."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from nema_forecast.config import DATA_CACHE_DIR, MODELS_DIR
from nema_forecast.data.iso_ne_ws import fetch_dayahead_demand_recent

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def run_backtest(days_back: int = 60) -> pd.DataFrame:
    """Load pre-computed test results + ISO day-ahead demand and build a comparison frame.

    The ISO-NE benchmark is the day-ahead hourly demand forecast for NEMA. It is sourced
    from the backfilled ``iso_forecast_history.parquet`` when present, otherwise fetched
    live for the last *days_back* days via the Web Services API.

    Returns a DataFrame with columns:
        datetime, actual, catboost_pred, iso_forecast
    """
    results_path = MODELS_DIR / "test_results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Run training first — missing {results_path}")

    cat_df = pd.read_parquet(results_path)
    cat_df["datetime"] = pd.to_datetime(cat_df["datetime"])

    # ISO-NE day-ahead demand benchmark — prefer the backfilled history, else live.
    history_path = DATA_CACHE_DIR / "iso_forecast_history.parquet"
    if history_path.exists():
        iso_df = pd.read_parquet(history_path)
        iso_df["datetime"] = pd.to_datetime(iso_df["datetime"])
    else:
        iso_df = fetch_dayahead_demand_recent(days_back=days_back)

    if iso_df.empty:
        logger.warning("No ISO day-ahead data found — backtest will be CatBoost-only")
        cat_df["iso_forecast"] = np.nan
        return cat_df

    iso_slim = iso_df[["datetime", "iso_forecast"]].copy()
    iso_slim["datetime"] = pd.to_datetime(iso_slim["datetime"])

    merged = cat_df.merge(iso_slim, on="datetime", how="left")
    matched = merged.dropna(subset=["iso_forecast"])
    logger.info(
        "Backtest: %d CatBoost rows, %d matched with ISO forecast",
        len(cat_df),
        len(matched),
    )
    return merged


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute summary metrics for CatBoost and ISO-NE."""
    result: dict = {}

    mask_cat = df["actual"].notna() & df["catboost_pred"].notna()
    if mask_cat.any():
        subset = df[mask_cat]
        result["catboost"] = {
            "MAE": round(mean_absolute_error(subset["actual"], subset["catboost_pred"]), 2),
            "MAPE": round(mean_absolute_percentage_error(subset["actual"], subset["catboost_pred"]) * 100, 2),
            "R2": round(r2_score(subset["actual"], subset["catboost_pred"]), 4),
            "n": int(mask_cat.sum()),
        }

    mask_iso = df["actual"].notna() & df["iso_forecast"].notna()
    if mask_iso.any():
        subset = df[mask_iso]
        result["iso"] = {
            "MAE": round(mean_absolute_error(subset["actual"], subset["iso_forecast"]), 2),
            "MAPE": round(mean_absolute_percentage_error(subset["actual"], subset["iso_forecast"]) * 100, 2),
            "R2": round(r2_score(subset["actual"], subset["iso_forecast"]), 4),
            "n": int(mask_iso.sum()),
        }

    return result


def compute_hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """MAE by hour-of-day for both models."""
    df = df.copy()
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

    rows = []
    for h in range(24):
        sub = df[df["hour"] == h]
        row: dict = {"hour": h}
        mask_c = sub["catboost_pred"].notna()
        if mask_c.any():
            row["catboost_mae"] = mean_absolute_error(sub.loc[mask_c, "actual"], sub.loc[mask_c, "catboost_pred"])
        mask_i = sub["iso_forecast"].notna()
        if mask_i.any():
            row["iso_mae"] = mean_absolute_error(sub.loc[mask_i, "actual"], sub.loc[mask_i, "iso_forecast"])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """MAE by month for both models."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["datetime"]).dt.month

    rows = []
    for m in sorted(df["month"].unique()):
        sub = df[df["month"] == m]
        row: dict = {"month": m}
        mask_c = sub["catboost_pred"].notna()
        if mask_c.any():
            row["catboost_mae"] = mean_absolute_error(sub.loc[mask_c, "actual"], sub.loc[mask_c, "catboost_pred"])
        mask_i = sub["iso_forecast"].notna()
        if mask_i.any():
            row["iso_mae"] = mean_absolute_error(sub.loc[mask_i, "actual"], sub.loc[mask_i, "iso_forecast"])
        rows.append(row)
    return pd.DataFrame(rows)


def save_backtest(df: pd.DataFrame, metrics: dict) -> None:
    out = MODELS_DIR / "backtest_results.parquet"
    df.to_parquet(out, index=False)
    (MODELS_DIR / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Backtest saved → %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CatBoost vs ISO-NE backtest")
    parser.add_argument("--days", type=int, default=60, help="Days of ISO forecasts to fetch")
    args = parser.parse_args()

    df = run_backtest(days_back=args.days)
    metrics = compute_metrics(df)
    save_backtest(df, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
