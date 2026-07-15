#!/usr/bin/env python
"""Stage-1 recovery figure -- SPEARMAN RHO companion to make_recovery_figures.py.

Same panel C layout (per-parameter bars, GRU embed=4/embed=2 vs baseline_rl at a chosen
N), but reporting Spearman rank correlation instead of R2. Rho is scale-invariant --
useful for the parameters (softmax_temp especially) where the fitted-value SCALE is
off enough to sink R2 even when the rank ordering across subjects is well recovered.
Panels A/B (fit-quality lines, mean-R2-vs-N) have no natural rho analog and are omitted;
this is a per-parameter bar-only companion, matching the "separate companion PNG" house
convention (paired report captions cite both plots together).

Inputs: same two CSVs as make_recovery_figures.py, now carrying spearman_{biasL,
learn_rate,softmax_temp} columns (backfilled 2026-07-15).
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = ["biasL", "learn_rate", "softmax_temp"]


def make_figure(gru, baseline, out_png, focus_n=200, hid_focus=16):
    gcol = {"h16e4": "#6baed6", "e2": "#7f7f7f", "base": "#000000"}
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    xpos = np.arange(3); w = 0.27
    g200 = gru[(gru.hid == hid_focus) & (gru.emb == 4) & (gru.subj == focus_n)].iloc[0]
    g200e2 = gru[(gru.hid == hid_focus) & (gru.emb == 2) & (gru.subj == focus_n)].iloc[0]
    b200 = baseline[baseline.num_subjects == focus_n].iloc[0]
    ax.bar(xpos - w, [b200[f"spearman_{p}"] for p in PARAMS], w, color="black", label="baseline_rl")
    ax.bar(xpos, [g200e2[f"spearman_{p}"] for p in PARAMS], w, color=gcol["e2"], label="GRU 2-d embedding")
    ax.bar(xpos + w, [g200[f"spearman_{p}"] for p in PARAMS], w, color=gcol["h16e4"], label="GRU 4-d embedding")
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels(PARAMS, fontsize=9); ax.set_ylabel("Spearman \u03c1")
    ax.set_ylim(0, 1.18); ax.set_title(f"Per-parameter rank correlation (n={focus_n})", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-csv", required=True)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--out", default="stage1_recovery_rho.png")
    a = ap.parse_args()
    gru = pd.read_csv(a.gru_csv); baseline = pd.read_csv(a.baseline_csv)
    make_figure(gru, baseline, a.out)
    print("wrote", a.out)
