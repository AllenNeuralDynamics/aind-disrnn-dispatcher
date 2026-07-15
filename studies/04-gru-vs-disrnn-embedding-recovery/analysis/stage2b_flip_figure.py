#!/usr/bin/env python
"""Stage-2b BASELINE-FLIP figure, house-styled to match stage-1/stage-2 recovery figures.

  a  Fit quality (likelihood relative to ground truth) at N=200, grouped by stage
     (S2 interpolation vs S2b held-out-tail non-monotonic drift), three models.
     The static baseline COLLAPSES under S2b extrapolation (0.939) while both GRUs stay
     >0.987 — the study's spine.
  b  Where the separation now lives: the S2b relative-LL gaps between models.

House convention (shared with stage 1/2): black = baseline, light blue = GRU 4-d
session-blind, dark blue = GRU 4-d + session conditioning. Square despined panels,
presentation-scale fonts, matplotlib default base + per-element overrides. Offline: reads
the committed curated stage2b_likelihood_flip.csv (frozen, keyed by wandb_run_id).

Single seed (42) per cell -> no error bars.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"baseline_rl": "black", "gru_session_blind": "#6baed6", "gru_session_conditioned": "#08519c"}
LAB = {"baseline_rl": "baseline Q-learning", "gru_session_blind": "GRU 4-d, session-blind",
       "gru_session_conditioned": "GRU 4-d + session cond."}
ORDER = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]


def make_figure(flip, out_png):
    fig = plt.figure(figsize=(9, 3.6)); gs = fig.add_gridspec(1, 2, wspace=0.42)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- a: fit quality by stage ----
    stages = ["S2", "S2b"]; xpos = np.arange(len(stages)); w = 0.27
    for i, m in enumerate(ORDER):
        vals = [flip[(flip.stage == s) & (flip.model == m)].rel_LL.iloc[0] for s in stages]
        bars = axA.bar(xpos + (i - 1) * w, vals, w, color=COL[m], label=LAB[m])
        for b, v in zip(bars, vals):
            axA.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=6.5)
    axA.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axA.set_xticks(xpos)
    axA.set_xticklabels(["Stage 2\n(interpolation)", "Stage 2b\n(held-out tail,\nnon-monotonic)"], fontsize=7)
    axA.set_ylabel("likelihood relative to ground truth"); axA.set_ylim(0.90, 1.02)
    axA.set_title("The baseline flip (N=200)", pad=18)
    axA.legend(frameon=False, loc="lower center", fontsize=5.5, ncol=3,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.0, handletextpad=0.4)

    # ---- b: S2b relative-LL gaps ----
    s2b = {r.model: r.rel_LL for r in flip[flip.stage == "S2b"].itertuples()}
    # gap bars colored by the condition ADDED (logical to the house palette)
    gaps = [("GRU session-blind\n\u2212 baseline", s2b["gru_session_blind"] - s2b["baseline_rl"], COL["gru_session_blind"]),
            ("GRU session-cond.\n\u2212 baseline", s2b["gru_session_conditioned"] - s2b["baseline_rl"], COL["gru_session_conditioned"]),
            ("GRU session-cond.\n\u2212 session-blind", s2b["gru_session_conditioned"] - s2b["gru_session_blind"], "#9ecae1")]
    yp = np.arange(len(gaps))[::-1]
    for y, (lab, g, c) in zip(yp, gaps):
        axB.barh(y, g, color=c); axB.text(g + 0.0012, y, f"+{g:.4f}", va="center", fontsize=6.5)
    axB.set_yticks(yp[::-1]); axB.set_yticklabels([g[0] for g in gaps][::-1], fontsize=6.5)
    axB.set_xlabel("relative-LL gap (Stage 2b, N=200)"); axB.set_xlim(0, 0.06)
    axB.set_title("Where model separation lives")

    for ax in (axA, axB):
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, l in zip((axA, axB), "ab"):
        ax.text(-0.18, 1.04, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-csv", default="stage2b_likelihood_flip.csv")
    ap.add_argument("--out", default="stage2b_likelihood_flip.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.flip_csv), a.out)
    print("wrote", a.out)
