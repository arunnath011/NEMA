"""Live track record — Beacon's locked forecasts vs actuals ISO-NE published *afterwards*.

This is the page that answers the honest skeptic: "how do I know you didn't just fit the curve
after the fact?" Every forecast here was committed to git before ISO-NE released the matching
zonal actual (they publish on a ~2-3 day lag), so the commit history is independent proof the
prediction came first. The record is forward-only — it starts the day it goes live and grows.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nema_forecast.dashboard.components import BLUE, GREEN
from nema_forecast.dashboard.live_data import maybe_update_and_load_ledger
from nema_forecast.forecast_ledger import summarize

ISO_ORANGE = "#E67E22"
SHADE = "rgba(230, 126, 34, 0.08)"
REPO_COMMITS_URL = "https://github.com/arunnath011/NEMA/commits/main/data/forecast_ledger.parquet"


def render() -> None:
    st.title("Live track record — locked forecasts vs reality")
    st.markdown(
        "Every point below is a forecast Beacon **committed to git before ISO New England "
        "published the actual demand** for that hour (ISO-NE releases NEMA zonal actuals on a "
        "~2-3 day lag). The actual is filled in only once it exists, so this can't be curve-"
        "fitting after the fact — the "
        f"[commit history]({REPO_COMMITS_URL}) timestamps each prediction independently. "
        "The record is **forward-only**: it started the day it went live and grows by a day "
        "each day — it is never back-filled from data already in hand."
    )

    with st.spinner("Checking for a new locked forecast …"):
        ledger = maybe_update_and_load_ledger()
    if ledger.empty:
        st.info("The track record has not been seeded yet — the first locked forecast appears after the next run.")
        return

    s = summarize(ledger)
    _headline(s)
    st.divider()
    _reveal_chart(ledger)

    scored = ledger.dropna(subset=["actual_mw"])
    if scored.empty:
        pend_to = s.get("latest_pending_target")
        st.info(
            "No forecasts have matured yet — ISO-NE publishes the actuals about 2-3 days after "
            "each forecast is locked"
            + (
                f", so the locked hours through **{pend_to:%b %d}** should be scored within a few days."
                if pend_to is not None
                else "."
            )
        )
    else:
        _daily_table(scored)

    with st.expander("How this is scored (and its limits)"):
        st.markdown(
            "- **Horizon:** each forecast is Beacon's **day-ahead (24 h)** prediction from the "
            "last actual available at forecast time — the same horizon as ISO-NE's day-ahead.\n"
            "- **Proof:** `data/forecast_ledger.parquet` is append-only; a locked `forecast_mw` "
            "is never modified, only its `actual_mw` is filled in later. Verify the timestamps in "
            f"the [git history]({REPO_COMMITS_URL}).\n"
            "- **Limit we're honest about:** target-hour *weather* comes from Open-Meteo's record "
            "for those hours, a touch better than the live weather forecast a real-time operator "
            "would have had. The *load* side is a genuine ahead-of-time prediction — its actual "
            "did not exist when the forecast was locked."
        )


def _headline(s: dict) -> None:
    c1, c2, c3 = st.columns(3)
    if "mae" in s:
        c1.metric("Live day-ahead MAE", f"{s['mae']:.0f} MW", help="Over matured (scored) hours only.")
    else:
        c1.metric("Live day-ahead MAE", "— MW", help="Appears once the first forecasts mature.")
    if "iso_mae" in s:
        delta = s["iso_mae"] - s["mae"]
        c2.metric(
            "ISO-NE MAE (same hours)",
            f"{s['iso_mae']:.0f} MW",
            delta=f"{delta:+.0f} MW vs Beacon",
            delta_color="inverse",
        )
    else:
        c2.metric("Forecast hours locked", f"{s['n_total']:,}")
    c3.metric("Scored / pending", f"{s['n_scored']} / {s['n_pending']}")

    made = s.get("latest_made_at")
    pend_to = s.get("latest_pending_target")
    if made is not None and pend_to is not None:
        st.caption(
            f"Latest forecast locked **{made:%b %d, %Y %H:%M} UTC**, covering hours through "
            f"**{pend_to:%b %d %H:%M}** — those actuals have not been published by ISO-NE yet."
        )


def _reveal_chart(ledger: pd.DataFrame) -> None:
    st.subheader("Forecast, and the actual as it is revealed")
    df = ledger.sort_values("target_datetime")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["target_datetime"],
            y=df["forecast_mw"],
            mode="lines",
            name="Beacon (locked forecast)",
            line={"color": GREEN, "width": 2.2},
        )
    )
    scored = df.dropna(subset=["actual_mw"])
    if not scored.empty:
        fig.add_trace(
            go.Scatter(
                x=scored["target_datetime"],
                y=scored["actual_mw"],
                mode="lines",
                name="Actual (revealed later)",
                line={"color": BLUE, "width": 2.6},
            )
        )
    if df["iso_forecast_mw"].notna().any():
        iso = df.dropna(subset=["iso_forecast_mw"])
        fig.add_trace(
            go.Scatter(
                x=iso["target_datetime"],
                y=iso["iso_forecast_mw"],
                mode="lines",
                name="ISO-NE (day-ahead)",
                line={"color": ISO_ORANGE, "width": 1.6, "dash": "dot"},
            )
        )

    # Shade the un-revealed region (target hours whose actual ISO-NE has not published yet).
    pending = df[df["actual_mw"].isna()]
    if not pending.empty:
        x0 = pending["target_datetime"].min()
        x1 = df["target_datetime"].max()
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=SHADE,
            line_width=0,
            annotation_text="awaiting ISO-NE actuals",
            annotation_position="top left",
            annotation={"font": {"color": ISO_ORANGE, "size": 12}},
        )

    fig.update_layout(
        yaxis_title="Load (MW)",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.1},
        height=460,
        margin={"t": 40, "b": 40, "l": 60, "r": 20},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _daily_table(scored) -> None:
    st.subheader("Scored days")
    d = scored.copy()
    d["day"] = d["target_datetime"].dt.date
    d["abs_err"] = (d["forecast_mw"] - d["actual_mw"]).abs()
    agg = {"abs_err": "mean", "actual_mw": "mean", "target_datetime": "count"}
    if d["iso_forecast_mw"].notna().any():
        d["iso_abs_err"] = (d["iso_forecast_mw"] - d["actual_mw"]).abs()
        agg["iso_abs_err"] = "mean"
    g = d.groupby("day").agg(agg).rename(columns={"target_datetime": "hours"})

    tbl = g.reset_index()
    tbl["Beacon MAE (MW)"] = tbl["abs_err"].round(0)
    tbl["Avg load (MW)"] = tbl["actual_mw"].round(0)
    tbl["Beacon MAPE"] = (tbl["abs_err"] / tbl["actual_mw"] * 100).round(1).astype(str) + "%"
    cols = {"day": "Day", "hours": "Hours"}
    show = tbl.rename(columns=cols)[["Day", "Hours", "Beacon MAE (MW)", "Avg load (MW)", "Beacon MAPE"]]
    if "iso_abs_err" in tbl.columns:
        show.insert(3, "ISO-NE MAE (MW)", tbl["iso_abs_err"].round(0).to_numpy())
    st.dataframe(show, use_container_width=True, hide_index=True)
