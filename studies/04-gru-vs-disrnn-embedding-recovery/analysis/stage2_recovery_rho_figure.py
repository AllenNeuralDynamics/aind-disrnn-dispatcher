#!/usr/bin/env python
"""Stage-2 recovery figure -- SPEARMAN RHO companion to stage2_recovery_figure.py panel c.

Same per-parameter bar grouping (baseline_rl / GRU session-blind / GRU session-cond.)
at n=200, reporting Spearman rank correlation instead of R2. Complements the R2 panel:
softmax_temp's R2 is winsorized (r2_softmax_temp) because of scale-sensitivity to a few
runaway fits, but rho -- being scale-invariant -- needs no such correction and is reported
raw for all three models.

Input: stage2_persession_recovery.csv (spearman_biasL/spearman_learn_rate/softmax_spearman
columns, backfilled 2026-07-15 -- softmax_spearman previously existed for baseline_rl only).
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = ["biasL", "learn_rate", "softmax_temp"]
COL = {"none": "#6baed6", "scalar": "#08519c", "base": "black"}
COL_MAP = {"biasL": "spearman_biasL", "learn_rate": "spearman_learn_rate", "softmax_temp": "softmax_spearman"}
BARS = [("baseline_rl", "base", "baseline_rl"),
        ("gru_session_blind", "none", "GRU 4-d, session-blind"),
        ("gru_session_conditioned", "scalar", "GRU 4-d + session cond.")]


def make_figure(ps, out_png, focus_n=200):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    xpos = np.arange(3); w = 0.27
    for i, (model, ck, lab) in enumerate(BARS):
        row = ps[(ps.model == model) & (ps.num_subjects == focus_n)].iloc[0]
        ax.bar(xpos + (i - 1) * w, [row[COL_MAP[p]] for p in PARAMS], w, color=COL[ck], label=lab)
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=9)
    ax.set_ylabel("Spearman \u03c1"); ax.set_ylim(0, 1.32)
    ax.set_title(f"Per-session per-parameter rank correlation (n={focus_n})", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persession-csv", default="stage2_persession_recovery.csv")
    ap.add_argument("--out", default="stage2_recovery_rho.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.persession_csv), a.out)
    print("wrote", a.out)
