"""End-to-end training pipeline.

Usage::

    python -m nema_forecast.model.train
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from nema_forecast.config import (
    CATBOOST_PARAMS,
    DATA_CACHE_DIR,
    HORIZON,
    LOOKBACK,
    MODELS_DIR,
    TRAIN_CUTOFF,
)
from nema_forecast.data.preprocessing import (
    apply_imputation,
    clean_columns,
    compute_imputation_stats,
    merge_load_weather,
    save_imputation_stats,
    temporal_split,
)
from nema_forecast.data.weather import load_historical_weather
from nema_forecast.features.engineering import (
    create_sequences,
    engineer_features,
    extract_lag_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

VAL_FRACTION = 0.15


def run_training() -> dict:
    """Execute the full training pipeline and persist artefacts."""

    # ------------------------------------------------------------------
    # 1. Load data — the backfilled ISO-NE Web Services demand history
    # ------------------------------------------------------------------
    now = datetime.now()
    load_history_path = DATA_CACHE_DIR / "load_history.parquet"
    if not load_history_path.exists():
        raise FileNotFoundError(
            f"Missing {load_history_path}. Run the backfill first:\n"
            "  python -m nema_forecast.scripts.backfill_load --start 2017-03-01"
        )
    logger.info("Loading ISO-NE demand history from %s …", load_history_path)
    load_df = pd.read_parquet(load_history_path)
    load_df["datetime"] = pd.to_datetime(load_df["datetime"])
    logger.info(
        "Loaded %d hourly observations (%s → %s)",
        len(load_df),
        load_df["datetime"].min(),
        load_df["datetime"].max(),
    )

    logger.info("Loading weather data …")
    weather_df = load_historical_weather(extend_to=now)

    # ------------------------------------------------------------------
    # 2. Merge & clean
    # ------------------------------------------------------------------
    merged = merge_load_weather(load_df, weather_df)
    merged = clean_columns(merged)

    train_raw, test_raw = temporal_split(merged, TRAIN_CUTOFF)
    logger.info("Train: %d rows | Test: %d rows", len(train_raw), len(test_raw))

    # ------------------------------------------------------------------
    # 3. Imputation (stats from train only)
    # ------------------------------------------------------------------
    stats = compute_imputation_stats(train_raw)
    save_imputation_stats(stats, MODELS_DIR / "imputation_stats.json")

    train_clean = apply_imputation(train_raw, stats)
    test_clean = apply_imputation(test_raw, stats)

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    train_feat = engineer_features(train_clean)
    test_feat = engineer_features(test_clean)
    logger.info("Train features: %s | Test features: %s", train_feat.shape, test_feat.shape)

    # ------------------------------------------------------------------
    # 5. Create sequences → lag features
    # ------------------------------------------------------------------
    X_train, Y_train, feat_cols, rtlo_idx = create_sequences(train_feat, LOOKBACK, HORIZON)
    X_test, Y_test, _, _ = create_sequences(test_feat, LOOKBACK, HORIZON)

    X_train_gb, gb_names = extract_lag_features(X_train, feat_cols, rtlo_idx)
    X_test_gb, _ = extract_lag_features(X_test, feat_cols, rtlo_idx)

    # ------------------------------------------------------------------
    # 6. Train / val split (last VAL_FRACTION of training data)
    # ------------------------------------------------------------------
    val_size = int(len(X_train_gb) * VAL_FRACTION)
    X_tr, X_val = X_train_gb[:-val_size], X_train_gb[-val_size:]
    Y_tr, Y_val = Y_train[:-val_size, 0], Y_train[-val_size:, 0]

    logger.info(
        "GB features: %d | Train: %d | Val: %d | Test: %d",
        len(gb_names),
        len(X_tr),
        len(X_val),
        len(X_test_gb),
    )

    # ------------------------------------------------------------------
    # 7. Train CatBoost
    # ------------------------------------------------------------------
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_tr, Y_tr, eval_set=(X_val, Y_val), use_best_model=True)

    # ------------------------------------------------------------------
    # 8. Evaluate
    # ------------------------------------------------------------------
    preds = model.predict(X_test_gb)
    Y_te = Y_test[:, 0]

    metrics = {
        "MAE": float(mean_absolute_error(Y_te, preds)),
        "MAPE": float(mean_absolute_percentage_error(Y_te, preds) * 100),
        "R2": float(r2_score(Y_te, preds)),
        "train_samples": len(X_tr),
        "val_samples": len(X_val),
        "test_samples": len(X_test_gb),
    }
    logger.info("MAE=%.2f MW | MAPE=%.2f%% | R²=%.4f", metrics["MAE"], metrics["MAPE"], metrics["R2"])

    # ------------------------------------------------------------------
    # 9. Persist artefacts
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))
    logger.info("Model saved → %s", model_path)

    (MODELS_DIR / "feature_names.json").write_text(json.dumps(gb_names, indent=2))
    (MODELS_DIR / "model_performance.json").write_text(json.dumps(metrics, indent=2))
    (MODELS_DIR / "feature_importance.json").write_text(
        json.dumps(
            dict(zip(gb_names, model.get_feature_importance().tolist())),
            indent=2,
        )
    )

    _save_test_results(test_feat, Y_te, preds, LOOKBACK)

    return metrics


def _save_test_results(
    test_feat: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lookback: int,
) -> None:
    """Save aligned actuals vs predictions as Parquet for the dashboard."""
    dates = pd.to_datetime(test_feat["datetime"].values)
    aligned_dates = dates[lookback : lookback + len(y_pred)]

    results = pd.DataFrame({"datetime": aligned_dates, "actual": y_true, "catboost_pred": y_pred})
    out = MODELS_DIR / "test_results.parquet"
    results.to_parquet(out, index=False)
    logger.info("Test results saved → %s (%d rows)", out, len(results))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NEMA CatBoost model")
    parser.parse_args()
    metrics = run_training()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
