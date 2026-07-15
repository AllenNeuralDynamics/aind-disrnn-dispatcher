#!/usr/bin/env python
"""Stage-2b recovery figure -- SPEARMAN RHO companion to stage2b_trajectory_figure.py panel a.

Same 3-condition grouping (baseline_rl / GRU subject-only / GRU session-conditioned) at
n=200, reporting Spearman rank correlation instead of R2. Stage 2 rows in the same CSV
are intentionally excluded (panel a of stage2b_trajectory_figure.py is Stage-2b-only, per
house scope; Stage 2's own rho companion is stage2_recovery_rho_figure.py).

Input: stage2b_trajectory_recovery.csv (spearman_biasL/spearman_learn_rate/
spearman_softmax_temp columns, backfilled 2026-07-15 via `STAGE=stage2b
stage2_session_traj.py` reconstruction + baseline_rl merge from stage2b_baseline_persession.csv).
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARAMS = ["biasL", "learn_rate", "softmax_temp"]
COL = {"baseline_rl": "black", "subj_only": "#6baed6", "conditioned": "#08519c"}
LAB = {"baseline_rl": "baseline_rl", "subj_only": "GRU 4-d, subject-only", "conditioned": "GRU 4-d + session cond."}
COL_MAP = {"biasL": "spearman_biasL", "learn_rate": "spearman_learn_rate", "softmax_temp": "spearman_softmax_temp"}
ORDER = ["baseline_rl", "subj_only", "conditioned"]


def make_figure(tr, out_png):
    s2b = tr[tr.stage == "S2b"]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    xpos = np.arange(3); w = 0.27
    for i, cond in enumerate(ORDER):
        row = s2b[s2b.cond == cond].iloc[0]
        ax.bar(xpos + (i - 1) * w, [row[COL_MAP[p]] for p in PARAMS], w, color=COL[cond], label=LAB[cond])
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=9)
    ax.set_ylabel("Spearman \u03c1"); ax.set_ylim(0, 1.32)
    ax.set_title("Stage 2b per-parameter rank correlation (n=200)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-csv", default="stage2b_trajectory_recovery.csv")
    ap.add_argument("--out", default="stage2b_recovery_rho.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.traj_csv), a.out)
    print("wrote", a.out)
