"""Locked forecast ledger — an append-only, git-committed record of Beacon's day-ahead
forecasts, written *before* ISO-NE publishes the corresponding actuals.

Why this exists: ISO-NE releases NEMA zonal real-time demand on a ~2-3 day lag, so any live
comparison is necessarily retrospective and *looks* like hindsight — a stranger can't tell a
genuine out-of-sample forecast from a curve fit after the fact. The ledger removes that doubt:
each forecast row is committed to git with the timestamp it was made, and the matching
``actual_mw`` is filled in only days later when ISO-NE publishes it. The commit history is
independent, tamper-evident proof the forecast predated the actual — a verifiable, growing
track record rather than a claim to be trusted.

Integrity rules (enforced here):
  * A target hour's ``forecast_mw`` / ``forecast_made_at`` are written once and never changed.
  * ``actual_mw`` (+ ``iso_forecast_mw``, ``scored_at``) are filled in later, exactly once.
  * The ledger is **forward-only**: it is never back-filled from data already in hand — doing
    so would manufacture the very hindsight it exists to disprove. It starts empty and grows by
    one day per scheduled run.
"""

from __future__ import annotations

import logging

import pandas as pd

from nema_forecast.config import LEDGER_PATH

logger = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "target_datetime",  # the forecasted hour (local, naive)
    "forecast_mw",  # Beacon's locked day-ahead prediction — never mutated
    "forecast_made_at",  # UTC time the row was first written (git commit ≈ this)
    "origin_datetime",  # last actual hour used as the forecast origin
    "horizon_h",  # target_datetime - origin_datetime, in hours
    "actual_mw",  # ISO-NE real-time actual, filled in once published (NaN until then)
    "iso_forecast_mw",  # ISO-NE's own day-ahead forecast for the hour (benchmark)
    "scored_at",  # UTC time the actual was filled in
]


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with the ledger schema, correct dtypes, deduped and sorted by target hour."""
    df = df.copy()
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["target_datetime"] = pd.to_datetime(df["target_datetime"])
    df["origin_datetime"] = pd.to_datetime(df["origin_datetime"])
    df["forecast_made_at"] = pd.to_datetime(df["forecast_made_at"], utc=True)
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    for col in ("forecast_mw", "actual_mw", "iso_forecast_mw", "horizon_h"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["target_datetime"], keep="first")
    return df[LEDGER_COLUMNS].sort_values("target_datetime").reset_index(drop=True)


def load_ledger() -> pd.DataFrame:
    """Load the committed ledger (empty, correctly-typed frame if it does not exist yet)."""
    if LEDGER_PATH.exists():
        return _coerce(pd.read_csv(LEDGER_PATH))
    return _coerce(pd.DataFrame(columns=LEDGER_COLUMNS))


def to_csv(df: pd.DataFrame) -> str:
    """Serialise the ledger to CSV text (the exact bytes committed to git)."""
    return _coerce(df).to_csv(index=False)


def save_ledger(df: pd.DataFrame) -> None:
    """Persist the ledger to ``LEDGER_PATH`` (CSV — readable diffs in git history)."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(to_csv(df))


def append_forecasts(ledger: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Add locked forecasts for target hours **not already present** — never overwrite.

    A target hour already in the ledger keeps its original ``forecast_mw`` / ``forecast_made_at``;
    this is what makes the record immutable and the git-timestamp proof meaningful.
    """
    ledger = _coerce(ledger)
    existing = set(pd.to_datetime(ledger["target_datetime"]))
    add = new_rows[~pd.to_datetime(new_rows["target_datetime"]).isin(existing)]
    if add.empty:
        logger.info("No new target hours to lock (all %d already in ledger).", len(new_rows))
        return ledger
    logger.info("Locking %d new forecast hours into the ledger.", len(add))
    return _coerce(pd.concat([ledger, add], ignore_index=True))


def score_pending(ledger: pd.DataFrame, actuals: pd.DataFrame, iso: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill ``actual_mw`` (+ ``iso_forecast_mw``, ``scored_at``) for matured, unscored rows.

    *actuals* is ISO-NE real-time demand ``[datetime, RTLO]``; *iso* is ISO-NE day-ahead demand
    ``[datetime, iso_forecast]``. Only rows whose target hour now has a published actual and are
    not yet scored are touched; ``forecast_mw`` is never modified.
    """
    ledger = _coerce(ledger)
    if actuals is not None and not actuals.empty:
        amap = (
            actuals.assign(datetime=pd.to_datetime(actuals["datetime"]))
            .drop_duplicates("datetime")
            .set_index("datetime")["RTLO"]
        )
        matured = ledger["actual_mw"].isna() & ledger["target_datetime"].map(amap).notna()
        if matured.any():
            ledger.loc[matured, "actual_mw"] = ledger.loc[matured, "target_datetime"].map(amap).to_numpy()
            ledger.loc[matured, "scored_at"] = pd.Timestamp.now(tz="UTC")
            logger.info("Scored %d newly-matured forecast hours.", int(matured.sum()))

    if iso is not None and not iso.empty:
        imap = (
            iso.assign(datetime=pd.to_datetime(iso["datetime"]))
            .drop_duplicates("datetime")
            .set_index("datetime")["iso_forecast"]
        )
        need = ledger["iso_forecast_mw"].isna() & ledger["target_datetime"].map(imap).notna()
        if need.any():
            ledger.loc[need, "iso_forecast_mw"] = ledger.loc[need, "target_datetime"].map(imap).to_numpy()
    return ledger


def summarize(ledger: pd.DataFrame) -> dict:
    """Headline stats over the scored portion of the ledger, for the dashboard/track record."""
    ledger = _coerce(ledger)
    scored = ledger.dropna(subset=["actual_mw"])
    out: dict = {
        "n_total": len(ledger),
        "n_scored": len(scored),
        "n_pending": int(ledger["actual_mw"].isna().sum()),
    }
    if not scored.empty:
        err = (scored["forecast_mw"] - scored["actual_mw"]).abs()
        out["mae"] = float(err.mean())
        out["first_scored_target"] = scored["target_datetime"].min()
        out["last_scored_target"] = scored["target_datetime"].max()
        iso_scored = scored.dropna(subset=["iso_forecast_mw"])
        if not iso_scored.empty:
            out["iso_mae"] = float((iso_scored["iso_forecast_mw"] - iso_scored["actual_mw"]).abs().mean())
    if ledger["forecast_made_at"].notna().any():
        out["earliest_made_at"] = ledger["forecast_made_at"].min()
        out["latest_made_at"] = ledger["forecast_made_at"].max()
    if ledger["actual_mw"].isna().any():
        pending = ledger[ledger["actual_mw"].isna()]
        out["next_reveal_target"] = pending["target_datetime"].min()
        out["latest_pending_target"] = pending["target_datetime"].max()
    return out
