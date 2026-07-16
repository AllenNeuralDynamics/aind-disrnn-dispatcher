#!/usr/bin/env python
"""Stage-3 per-session recovery -- SPEARMAN RHO companion to stage3_recovery_vs_baseline.py
panel c.

Same horizontal-bar layout (family:param rows, 3 conditions: baseline_rl / GRU session-blind
/ GRU session-conditioned), reporting Spearman rank correlation instead of R2. Complements
the R2 panel where several parameters (RescorlaWagner biasL/learn_rate/epsilon, Bari2019
forget_rate_unchosen, etc.) show strongly negative R2 despite the estimator still tracking
subject RANK correctly (rho 0.2-0.9) -- see the report's R2-vs-rho discussion.

Input: stage3_persession_recovery.csv (spearman column, backfilled 2026-07-15 for the two
GRU conditions -- previously a np.nan placeholder in compute_stage3_persession_recovery.py;
baseline_rl already had it).
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"baseline_rl": "black", "gru_session_blind": "#6baed6", "gru_session_conditioned": "#08519c"}
LAB = {"baseline_rl": "baseline_rl", "gru_session_blind": "GRU (session-blind)",
       "gru_session_conditioned": "GRU (session-cond.)"}
FAM_SHORT = {"Bari2019": "Bari", "Hattori2019": "Hatt.", "RescorlaWagner": "RW"}
FAMS = ["Bari2019", "Hattori2019", "RescorlaWagner"]
CONDS = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]


def make_figure(ps, out_png):
    rows = []
    for fam in FAMS:
        params = sorted(ps[ps.true_preset == fam].param.unique())
        for p in params:
            rows.append((fam, p))
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 6.4))
    for i, cond in enumerate(CONDS):
        vals = []
        for fam, p in rows:
            sub = ps[(ps.true_preset == fam) & (ps.cond == cond) & (ps.param == p)]
            vals.append(sub.spearman.iloc[0] if len(sub) else np.nan)
        ax.barh(y - (i - 1) * 0.27, vals, 0.27, color=COL[cond], label=LAB[cond])
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.axvline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{FAM_SHORT[f]}: {p}" for f, p in rows], fontsize=8)
    ax.set_xlabel("per-session recovery Spearman \u03c1")
    ax.set_title("Stage 3 \u2014 per-session rank correlation, per parameter", fontsize=11, loc="left")
    ax.set_xlim(-0.25, 1.05)
    ax.set_ylim(y.min() - 1.1, y.max() + 0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persession-csv", default="stage3_persession_recovery.csv")
    ap.add_argument("--out", default="stage3_recovery_rho.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.persession_csv), a.out)
    print("wrote", a.out)
