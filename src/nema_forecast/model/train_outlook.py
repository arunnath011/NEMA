"""Train the calendar-future **Outlook** model — one pooled long-horizon CatBoost.

The day-ahead models (h=1..24) predict from fresh load, but NEMA zonal actuals lag ~3 days, so
to forecast *calendar-future* dates (today → +4 days) the model must tolerate that gap. The
backtest showed error is essentially flat (~100 MW) from ~72h out to +4 days, because past the
one-week lag window the forecast is driven by weather + calendar, which don't decay with lead.

So instead of ~150 per-hour models we train a **single pooled model** over a set of leads with
``lead_h`` (hours from the last known load to the target) as an explicit feature. It predicts at
any lead in range and stays compact (one artifact). Feature vector, in order:
``[lag features, target-hour exog (weather forecast + calendar), lead_h]`` — identical at serve.

    python -m nema_forecast.model.train_outlook
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from nema_forecast.config import CATBOOST_PARAMS, LOOKBACK, MODELS_DIR, TRAIN_CUTOFF
from nema_forecast.data.open_meteo import fetch_archive_weather
from nema_forecast.data.preprocessing import (
    apply_imputation,
    clean_columns,
    compute_imputation_stats,
    merge_load_weather,
    temporal_split,
)
from nema_forecast.features.engineering import (
    create_sequences,
    engineer_features,
    extract_lag_features,
    extract_target_exog,
    target_exog_names,
)
from nema_forecast.model.train import _load_load_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Leads (hours from last known load to target) to pool. 24→168h spans the whole calendar-future
# range we serve (given the ~2-3 day ISO lag, 168h ≈ +4 calendar days). The plateau means the
# model interpolates cleanly to any intermediate lead.
TRAIN_LEADS = [24, 48, 72, 96, 120, 144, 168]
H_MAX = max(TRAIN_LEADS)
OUTLOOK_MODEL_PATH = MODELS_DIR / "catboost_outlook.cbm"
OUTLOOK_META_PATH = MODELS_DIR / "outlook_meta.json"
VAL_FRACTION = 0.15


def _pooled(Xg, vals, feat_cols, Y, leads):
    """Stack (sample x lead) rows: [lag feats, target-exog(lead), lead_h] with targets Y[:,h]."""
    xs, ys = [], []
    for lead in leads:
        h = lead - 1
        exog = extract_target_exog(vals, feat_cols, h, LOOKBACK, H_MAX)
        lead_col = np.full((len(Xg), 1), float(lead))
        xs.append(np.hstack([Xg, exog, lead_col]))
        ys.append(Y[:, h])
    return np.vstack(xs), np.concatenate(ys)


def run() -> dict:
    load_df = _load_load_series()
    load_df = load_df.dropna(subset=["RTLO"]).drop_duplicates("datetime").sort_values("datetime")
    wstart = pd.to_datetime(load_df["datetime"].min()).strftime("%Y-%m-%d")
    wend = pd.to_datetime(load_df["datetime"].max()).strftime("%Y-%m-%d")
    weather_df = fetch_archive_weather(wstart, wend)

    merged = clean_columns(merge_load_weather(load_df, weather_df))
    train_raw, test_raw = temporal_split(merged, TRAIN_CUTOFF)
    stats = compute_imputation_stats(train_raw)
    train_feat = engineer_features(apply_imputation(train_raw, stats))
    test_feat = engineer_features(apply_imputation(test_raw, stats))

    feat_cols = [c for c in train_feat.columns if c != "datetime"]
    rtlo_idx = feat_cols.index("RTLO")
    X_tr, Y_tr, _, _ = create_sequences(train_feat, LOOKBACK, H_MAX)
    X_te, Y_te, _, _ = create_sequences(test_feat, LOOKBACK, H_MAX)
    Xg_tr, gb_names = extract_lag_features(X_tr, feat_cols, rtlo_idx)
    Xg_te, _ = extract_lag_features(X_te, feat_cols, rtlo_idx)
    del X_tr, X_te
    tr_vals = train_feat[feat_cols].values.astype(float)
    te_vals = test_feat[feat_cols].values.astype(float)

    x_tr, y_tr = _pooled(Xg_tr, tr_vals, feat_cols, Y_tr, TRAIN_LEADS)
    logger.info("Pooled training rows: %d over leads %s", len(x_tr), TRAIN_LEADS)

    val = max(1, int(len(x_tr) * VAL_FRACTION))
    # Shuffle so the held-out val split isn't a single lead/tail; keeps early stopping honest.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x_tr))
    x_tr, y_tr = x_tr[perm], y_tr[perm]

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(x_tr[:-val], y_tr[:-val], eval_set=(x_tr[-val:], y_tr[-val:]), use_best_model=True)
    model.save_model(str(OUTLOOK_MODEL_PATH))

    aug_names = gb_names + target_exog_names(feat_cols) + ["lead_h"]
    meta = {
        "lag_feature_names": gb_names,
        "exog_names": target_exog_names(feat_cols),
        "aug_feature_names": aug_names,
        "feat_cols": feat_cols,
        "lookback": LOOKBACK,
        "train_leads": TRAIN_LEADS,
        "min_lead": min(TRAIN_LEADS),
        "max_lead": max(TRAIN_LEADS),
    }
    OUTLOOK_META_PATH.write_text(json.dumps(meta, indent=2))

    # Held-out accuracy per lead (sanity: should sit near the ~100 MW backtest plateau).
    per_lead = {}
    for lead in TRAIN_LEADS:
        h = lead - 1
        exog = extract_target_exog(te_vals, feat_cols, h, LOOKBACK, H_MAX)
        x = np.hstack([Xg_te, exog, np.full((len(Xg_te), 1), float(lead))])
        pred = model.predict(x)
        mae = float(mean_absolute_error(Y_te[:, h], pred))
        r2 = float(r2_score(Y_te[:, h], pred))
        per_lead[lead] = {"mae": round(mae, 1), "r2": round(r2, 4)}
        logger.info("held-out lead %3dh | MAE %.1f | R2 %.3f", lead, mae, r2)

    meta["heldout_by_lead"] = per_lead
    OUTLOOK_META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s and %s", OUTLOOK_MODEL_PATH, OUTLOOK_META_PATH)
    return per_lead


if __name__ == "__main__":
    run()
