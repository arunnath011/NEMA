"""End-to-end training pipeline — direct multi-horizon Beacon model.

Trains one CatBoost model **per forecast horizon** (h=1..24). Each model predicts its
horizon directly from the 168 h lookback window, so the 24-hour forecast no longer relies
on recursive roll-out (which compounds error). The h=1 model is also saved as
``catboost_model.cbm`` for backward compatibility (hindcast / single-step callers).

Usage::

    python -m nema_forecast.model.train
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from nema_forecast.config import (
    CATBOOST_PARAMS,
    DATA_CACHE_DIR,
    HORIZON,
    LEGACY_DATA_PATH,
    LOOKBACK,
    MODELS_DIR,
    TRAIN_CUTOFF,
)
from nema_forecast.data.open_meteo import fetch_archive_weather
from nema_forecast.data.preprocessing import (
    apply_imputation,
    clean_columns,
    compute_imputation_stats,
    merge_load_weather,
    save_imputation_stats,
    temporal_split,
)
from nema_forecast.features.engineering import (
    create_sequences,
    engineer_features,
    extract_lag_features,
    extract_target_exog,
    target_exog_names,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

VAL_FRACTION = 0.15


def horizon_model_path(h: int) -> Path:
    """Path for the per-horizon model (h is 1-based)."""
    return MODELS_DIR / f"catboost_h{h:02d}.cbm"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_load_series() -> pd.DataFrame:
    """Load the NEMA hourly load series as ``[datetime, RTLO]``.

    Prefers the ISO-NE backfill (``load_history.parquet``); falls back to the legacy
    WHLSECOST CSVs under ``NEMA_DATA_PATH/{train,test}`` that produced the original model.
    """
    history = DATA_CACHE_DIR / "load_history.parquet"
    if history.exists():
        logger.info("Loading load series from %s", history)
        df = pd.read_parquet(history)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df[["datetime", "RTLO"]]

    from nema_forecast.data.iso_ne import _parse_whlsecost_csv

    logger.info("load_history.parquet not found — loading legacy WHLSECOST CSVs from %s", LEGACY_DATA_PATH)
    frames: list[pd.DataFrame] = []
    for sub in ("train", "test"):
        for f in sorted(glob.glob(str(LEGACY_DATA_PATH / sub / "whlsecost_hourly_4008_*.csv"))):
            part = _parse_whlsecost_csv(Path(f))
            if not part.empty:
                frames.append(part[["datetime", "RTLO"]])
    if not frames:
        raise FileNotFoundError(
            "No load data found. Provide data/cache/load_history.parquet (run the backfill) "
            "or set NEMA_DATA_PATH to the legacy WHLSECOST CSVs."
        )
    return pd.concat(frames).drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def run_training() -> dict:
    """Execute the direct multi-horizon training pipeline and persist artefacts."""
    # 1. Load + merge data --------------------------------------------------
    load_df = _load_load_series()
    logger.info("Load: %d h (%s → %s)", len(load_df), load_df["datetime"].min(), load_df["datetime"].max())

    # Weather from Open-Meteo archive — the SAME source used at serving (no train/serve
    # weather-source mismatch). Covers the load's date range.
    wstart = pd.to_datetime(load_df["datetime"].min()).strftime("%Y-%m-%d")
    wend = pd.to_datetime(load_df["datetime"].max()).strftime("%Y-%m-%d")
    logger.info("Loading Open-Meteo archive weather %s → %s …", wstart, wend)
    weather_df = fetch_archive_weather(wstart, wend)

    merged = clean_columns(merge_load_weather(load_df, weather_df))
    train_raw, test_raw = temporal_split(merged, TRAIN_CUTOFF)
    logger.info("Train: %d rows | Test: %d rows", len(train_raw), len(test_raw))

    # 2. Imputation (train stats only) -------------------------------------
    stats = compute_imputation_stats(train_raw)
    save_imputation_stats(stats, MODELS_DIR / "imputation_stats.json")
    train_clean = apply_imputation(train_raw, stats)
    test_clean = apply_imputation(test_raw, stats)

    # 3. Features + sequences (Y holds all HORIZON steps) ------------------
    train_feat = engineer_features(train_clean)
    test_feat = engineer_features(test_clean)
    X_train, Y_train, feat_cols, rtlo_idx = create_sequences(train_feat, LOOKBACK, HORIZON)
    X_test, Y_test, _, _ = create_sequences(test_feat, LOOKBACK, HORIZON)
    X_train_gb, gb_names = extract_lag_features(X_train, feat_cols, rtlo_idx)
    X_test_gb, _ = extract_lag_features(X_test, feat_cols, rtlo_idx)

    # Target-hour exogenous features (calendar + forecast weather at t+h, per horizon).
    train_vals = train_feat[feat_cols].values.astype(float)
    test_vals = test_feat[feat_cols].values.astype(float)
    exog_names = target_exog_names(feat_cols)
    aug_names = gb_names + exog_names

    val_size = max(1, int(len(X_train_gb) * VAL_FRACTION))
    logger.info(
        "Lag features: %d | target-exog: %d | Train: %d | Val: %d | Test: %d",
        len(gb_names),
        len(exog_names),
        len(X_train_gb) - val_size,
        val_size,
        len(X_test_gb),
    )

    # 4. Train one model per horizon (augmented features) ------------------
    direct_mae: list[float] = []
    single_mae: list[float] = []  # lags-only h=1 model applied to every horizon (naive baseline)

    # Lags-only h=1 baseline (one fit) for the horizon-degradation comparison.
    h1_lags = CatBoostRegressor(**CATBOOST_PARAMS)
    h1_lags.fit(
        X_train_gb[:-val_size],
        Y_train[:-val_size, 0],
        eval_set=(X_train_gb[-val_size:], Y_train[-val_size:, 0]),
        use_best_model=True,
    )
    h1_lags_test_pred = h1_lags.predict(X_test_gb)

    h1_aug_test_pred = np.zeros(len(X_test_gb))
    importance_model = h1_lags
    for h in range(HORIZON):
        e_tr = extract_target_exog(train_vals, feat_cols, h, LOOKBACK, HORIZON)
        e_te = extract_target_exog(test_vals, feat_cols, h, LOOKBACK, HORIZON)
        x_tr_h = np.hstack([X_train_gb, e_tr])
        x_te_h = np.hstack([X_test_gb, e_te])

        model = CatBoostRegressor(**CATBOOST_PARAMS)
        model.fit(
            x_tr_h[:-val_size],
            Y_train[:-val_size, h],
            eval_set=(x_tr_h[-val_size:], Y_train[-val_size:, h]),
            use_best_model=True,
        )
        model.save_model(str(horizon_model_path(h + 1)))

        direct_mae.append(float(mean_absolute_error(Y_test[:, h], model.predict(x_te_h))))
        single_mae.append(float(mean_absolute_error(Y_test[:, h], h1_lags_test_pred)))
        if h == 0:
            h1_aug_test_pred = model.predict(x_te_h)
            importance_model = model
            model.save_model(str(MODELS_DIR / "catboost_model.cbm"))  # backward compat (augmented h=1)
        logger.info(
            "h=%2d | direct(+weather) MAE %6.1f | naive single-model MAE %6.1f", h + 1, direct_mae[h], single_mae[h]
        )

    h1_model = importance_model
    h1_test_pred = h1_aug_test_pred

    # 5. Headline metrics (averaged across the 24-hour horizon) ------------
    direct_preds_h1 = h1_test_pred
    metrics = {
        "MAE": float(np.mean(direct_mae)),
        "MAE_h1": direct_mae[0],
        "MAE_h24": direct_mae[-1],
        "MAPE": float(mean_absolute_percentage_error(Y_test[:, 0], direct_preds_h1) * 100),
        "R2": float(r2_score(Y_test[:, 0], direct_preds_h1)),
        "horizon_degradation": direct_mae[-1] / direct_mae[0],
        "avg_improvement_vs_single_pct": float((np.mean(single_mae) - np.mean(direct_mae)) / np.mean(single_mae) * 100),
        "test_samples": len(X_test_gb),
        "n_horizon_models": HORIZON,
    }
    logger.info(
        "Direct multi-horizon: avg MAE %.1f (h1 %.1f → h24 %.1f) | %.1f%% better than single-model across 24h",
        metrics["MAE"],
        metrics["MAE_h1"],
        metrics["MAE_h24"],
        metrics["avg_improvement_vs_single_pct"],
    )

    # 6. Persist artefacts --------------------------------------------------
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(aug_names, indent=2))
    (MODELS_DIR / "model_performance.json").write_text(json.dumps(metrics, indent=2))
    (MODELS_DIR / "horizon_mae.json").write_text(
        json.dumps(
            {"horizon": list(range(1, HORIZON + 1)), "direct_mae": direct_mae, "single_model_mae": single_mae},
            indent=2,
        )
    )
    (MODELS_DIR / "feature_importance.json").write_text(
        json.dumps(dict(zip(aug_names, h1_model.get_feature_importance().tolist())), indent=2)
    )
    _save_test_results(test_feat, Y_test[:, 0], direct_preds_h1, LOOKBACK)
    return metrics


def _save_test_results(test_feat: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, lookback: int) -> None:
    """Save aligned h=1 actuals vs predictions as Parquet for the dashboard."""
    dates = pd.to_datetime(test_feat["datetime"].values)
    aligned_dates = dates[lookback : lookback + len(y_pred)]
    results = pd.DataFrame({"datetime": aligned_dates, "actual": y_true, "catboost_pred": y_pred})
    out = MODELS_DIR / "test_results.parquet"
    results.to_parquet(out, index=False)
    logger.info("Test results saved → %s (%d rows)", out, len(results))


def main() -> None:
    argparse.ArgumentParser(description="Train the direct multi-horizon Beacon model").parse_args()
    print(json.dumps(run_training(), indent=2))


if __name__ == "__main__":
    main()
