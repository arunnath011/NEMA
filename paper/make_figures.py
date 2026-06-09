"""Generate the paper figures from the trained-model artefacts (no network).

Outputs to paper/figures/:
  fig_horizon_mae.png  — per-horizon MAE, Beacon (direct) vs single-model roll-out
  fig_ablation.png     — day-ahead (h=24) MAE across the three cumulative configurations
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

GREEN, GREY = "#1E8449", "#7F8C8D"
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})


def fig_horizon_mae() -> None:
    hm = json.loads((ROOT / "models" / "horizon_mae.json").read_text())
    h, direct, single = hm["horizon"], hm["direct_mae"], hm["single_model_mae"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(h, single, "--o", color=GREY, ms=3, label="Single model (rolled out)")
    ax.plot(h, direct, "-o", color=GREEN, ms=3, lw=2.2, label="Beacon (per-horizon + target weather)")
    ax.set_xlabel("Forecast horizon $h$ (hours ahead)")
    ax.set_ylabel("Test MAE (MW)")
    ax.set_xlim(1, 24)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIGDIR / "fig_horizon_mae.png")
    plt.close(fig)


def fig_ablation() -> None:
    # FACTS.md §8.1 — cumulative day-ahead (h=24) MAE
    labels = ["Lags only\n(OWM)", "+ target weather\n(OWM)", "+ Open-Meteo\n(matched)"]
    h24 = [183.0, 110.3, 76.7]
    h1 = [78.7, 77.7, 60.9]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar([i - 0.2 for i in x], h24, width=0.4, color=GREEN, label="MAE @ h=24 (day-ahead)")
    ax.bar([i + 0.2 for i in x], h1, width=0.4, color=GREY, label="MAE @ h=1")
    for i, v in enumerate(h24):
        ax.text(i - 0.2, v + 3, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MAE (MW)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIGDIR / "fig_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_horizon_mae()
    fig_ablation()
    print("Wrote figures to", FIGDIR)
