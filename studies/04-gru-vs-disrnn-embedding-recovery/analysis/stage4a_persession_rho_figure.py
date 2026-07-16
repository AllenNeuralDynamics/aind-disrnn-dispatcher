#!/usr/bin/env python
"""Stage-4a per-session recovery -- SPEARMAN RHO companion to stage4a_persession_figure.py
(both panels a and b).

Panel a: mean per-session rank correlation over each family's own drifting parameter set
(same STATIC_PARAMS exclusion as the R2 panel: CompareToThreshold's static `threshold` is
excluded from the mean here, scored only in panel b).
Panel b: per-parameter rank correlation, horizontal bars grouped by family.

Input: stage4a_persession_recovery.csv (spearman column, backfilled 2026-07-15 for the two
GRU conditions -- previously a np.nan placeholder in compute_stage4a_persession_recovery.py;
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
FAM_SHORT = {"QLearning": "QL", "CompareToThreshold": "CTT", "LossCounting": "LC"}
FAMS = ["QLearning", "CompareToThreshold", "LossCounting"]
CONDS = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]
STATIC_PARAMS = {"threshold"}
P_SHORT = {"softmax_inverse_temperature": "softmax_temp", "loss_count_threshold_mean": "loss_thresh_mean"}


def _panel_a(ax, ps):
    xpos = np.arange(len(FAMS)); w = 0.27
    for i, cond in enumerate(CONDS):
        vals = []
        for fam in FAMS:
            sub = ps[(ps.true_family == fam) & (ps.cond == cond) & (~ps.param.isin(STATIC_PARAMS))]
            vals.append(sub.spearman.mean() if len(sub) else np.nan)
        ax.bar(xpos + (i - 1) * w, vals, w, color=COL[cond], label=LAB[cond])
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels([FAM_SHORT[f] for f in FAMS])
    ax.set_ylabel("mean per-session rank correlation \u03c1")
    ax.set_title("Per-session recovery,\nmean over family's drifting params", fontsize=13, loc="left")
    ax.set_ylim(0, 1.32)
    ax.legend(frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1)


def _panel_b(ax, ps):
    rows = []
    for fam in FAMS:
        params = sorted(ps[ps.true_family == fam].param.unique())
        for p in params:
            rows.append((fam, p))
    y = np.arange(len(rows))[::-1]
    for i, cond in enumerate(CONDS):
        vals = []
        for fam, p in rows:
            sub = ps[(ps.true_family == fam) & (ps.cond == cond) & (ps.param == p)]
            vals.append(sub.spearman.iloc[0] if len(sub) else np.nan)
        ax.barh(y - (i - 1) * 0.27, vals, 0.27, color=COL[cond], label=LAB[cond])
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.axvline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{FAM_SHORT[f]}: {P_SHORT.get(p, p)}" for f, p in rows], fontsize=11)
    ax.set_xlabel("per-session recovery Spearman \u03c1")
    ax.set_title("Per-session recovery, per parameter", fontsize=13, loc="left")
    ax.set_xlim(0, 1.05)


def make_figure(ps, out_png):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.7]})
    _panel_a(axA, ps)
    _panel_b(axB, ps)
    axA.set_box_aspect(1)
    for a in (axA, axB):
        for s in a.spines.values():
            s.set_visible(False)
    for a, l in zip((axA, axB), "ab"):
        a.text(-0.16, 1.08, l, transform=a.transAxes, fontsize=17, fontweight="bold", va="bottom")
    fig.suptitle("Stage 4a \u2014 mixture of model families: per-session rank correlation, baseline vs GRU",
                 fontsize=16, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.24, wspace=0.45)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persession-csv", default="stage4a_persession_recovery.csv")
    ap.add_argument("--out", default="stage4a_persession_rho.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.persession_csv), a.out)
    print("wrote", a.out)
