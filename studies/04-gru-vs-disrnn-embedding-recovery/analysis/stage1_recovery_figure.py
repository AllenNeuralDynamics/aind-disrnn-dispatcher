#!/usr/bin/env python3
"""Stage-1 parameter-recovery figure (preliminary, embedding-size sweep).

OFFLINE single producer. Reads two committed curated CSVs and writes one figure:

  inputs  (committed, in this directory):
    stage1_recovery_grid.csv          per (n_subjects, embedding_size, param) recovery R²,
                                      one deterministic row per cell at hidden_size=16, seed=42
                                      (columns wandb_run_id + artifact_version_id pin the exact run)
    stage1_recovery_scatter_n200_e4.csv   true vs recovered per subject, 200-subject / embed-4 cell

Each grid cell is a SINGLE run (the Stage-1 sweep used seed=42 only), so there are no
seed replicates and hence no error bars. hidden_size is fixed at 16 (the settled Stage-1
config); the embed-4 result is width-insensitive, while embed-2 sits below the
identifiability threshold — that gap, not run-to-run noise, is the point of panel a.

  output:
    figures/stage1_recovery_preliminary.png

Panel a: recovery R² vs cohort size, embedding size 4 (solid) vs 2 (dashed).
Panel b: recovered-vs-true scatter for the settled 200-subject / embed-4 / hidden-16
         cell, one square subplot per recoverable parameter with the y=x identity line.

Recovery method (both panels): per parameter, 5-fold cross-validated Ridge (alpha=1)
regression from the standardized subject-embedding vector to the ground-truth subject
parameter; score = out-of-fold R². Ground-truth subject value = the subject's session
constant (Stage-1 data is static, so all sessions share one parameter draw).

Needs only numpy / pandas / matplotlib. Numbers are frozen in the CSVs; to refresh them
from W&B, re-run the extraction on an authenticated HPC node (see study README) and
overwrite the CSVs -- this script never calls the network.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
PCOL = {"biasL": "#4C72B0", "learn_rate": "#DD8452", "softmax_temp": "#55A868"}
PARAMS = ["biasL", "learn_rate", "softmax_temp"]


def _style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 17,
        "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 2.4, "lines.markersize": 8, "savefig.dpi": 200,
    })


def _panel_a(ax, grid):
    for sname in PARAMS:
        for emb, ls, al, lw in [(4, "-", 1.0, 2.6), (2, "--", 0.55, 1.8)]:
            sub = grid[(grid.embedding_size == emb) & (grid.param == sname)].sort_values("n_subjects")
            ax.plot(sub.n_subjects, sub.recovery_r2, ls, color=PCOL[sname],
                    alpha=al, lw=lw, marker="o", ms=8)
    ax.axhline(1.0, color="0.6", lw=1.0, ls=":", zorder=0)
    ax.set_xlabel("# subjects")
    ax.set_ylabel("recovery R²  (embedding → true param)")
    ax.set_title("Parameter recovery: embedding size 4 (solid) vs 2 (dashed)", loc="left")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(sorted(grid.n_subjects.unique()))
    ax.margins(x=0.22)
    xmax = grid.n_subjects.max()
    for sname, yy in {"learn_rate": 1.02, "softmax_temp": 0.90, "biasL": 0.80}.items():
        ax.annotate(sname, (xmax, yy), xytext=(8, 0), textcoords="offset points",
                    va="center", color=PCOL[sname], fontsize=14, weight="bold")


def _panel_b(axes, scat):
    from numpy import polyfit  # noqa: F401  (kept for parity; not used)
    for k, sname in enumerate(PARAMS):
        t = scat[f"{sname}_true"].to_numpy(float)
        pr = scat[f"{sname}_recovered"].to_numpy(float)
        r2 = 1.0 - np.sum((t - pr) ** 2) / np.sum((t - t.mean()) ** 2)
        ax = axes[k]
        ax.scatter(t, pr, s=16, color=PCOL[sname], alpha=0.55, edgecolor="none")
        lo = min(t.min(), pr.min()); hi = max(t.max(), pr.max()); pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=1.0, alpha=0.5, zorder=0)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_box_aspect(1)  # square: identity line reads at 45°
        ax.set_title(f"{sname}\nR²={r2:.2f}", fontsize=13)
        ax.set_xlabel("true"); ax.set_ylabel("recovered")


def main():
    _style()
    grid = pd.read_csv(os.path.join(HERE, "stage1_recovery_grid.csv"))
    scat = pd.read_csv(os.path.join(HERE, "stage1_recovery_scatter_n200_e4.csv"))
    n_b = len(scat)

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(1, 4, wspace=0.6)
    axA = fig.add_subplot(gs[0, 0:2])
    gsB = gs[0, 2:4].subgridspec(1, 3, wspace=0.65)
    axB = [fig.add_subplot(gsB[0, k]) for k in range(3)]

    _panel_a(axA, grid)
    _panel_b(axB, scat)
    fig.text(0.72, 1.0, f"Recovered vs true (n={n_b}, hidden=16, embed=4)",
             ha="center", va="top", fontsize=15)
    axA.text(-0.14, 1.10, "a", transform=axA.transAxes, fontsize=18, weight="bold")
    axB[0].text(-0.42, 1.18, "b", transform=axB[0].transAxes, fontsize=18, weight="bold")

    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "stage1_recovery_preliminary.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
