#!/usr/bin/env python
"""Stage-2b SESSION-TRAJECTORY recovery (panels a,b), house-styled to match stage 1/2.

  a  Per-session PARAMETER recovery R2 by parameter: Stage-2 vs Stage-2b, subject-only
     (session-blind) vs session-conditioned. Stronger non-monotonic drift (S2b) lowers
     recovery, and the session-conditioning delta adds the most where drift is largest.
  b  Session-POSITION recovery (sessfrac R2): how well the embedding locates a session
     within its subject. Subject-only is 0.00 by construction (a fixed per-subject point
     carries no session index); session conditioning recovers it (S2 0.94 -> S2b 0.47 as
     the drift turns non-monotonic).

House convention: color = session-condition (light blue = subject-only/session-blind,
dark blue = session-conditioned); stage distinguished by saturation (Stage-2 lighter,
Stage-2b full). Square despined panels, presentation-scale fonts. Offline: reads the
committed curated stage2b_trajectory_recovery.csv (frozen, from the session-trajectory
recovery CSVs). Single seed (42) per cell -> no error bars.

NOTE: the qualitative drift-paths panel (embedding trajectories) from the original figure
is omitted here because it needs per-session embedding coordinates that are not committed
(they require a W&B pull); this producer stays fully offline.
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = {"subj_only": "#6baed6", "conditioned": "#08519c"}
ALPHA = {"S2": 0.55, "S2b": 1.0}
PARAMS = ["biasL", "learn_rate", "softmax_temp"]


def _bars4(ax, df, valfn, ylabel, title, annotate_zero=False):
    xpos = np.arange(len(PARAMS)) if valfn is None else None
    combos = [("S2", "subj_only"), ("S2", "conditioned"), ("S2b", "subj_only"), ("S2b", "conditioned")]
    w = 0.2
    for i, (stage, cond) in enumerate(combos):
        row = df[(df.stage == stage) & (df.cond == cond)].iloc[0]
        vals = [row[f"r2_{p}"] for p in PARAMS]
        b = ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, color=BLUE[cond], alpha=ALPHA[stage],
                   label=f"{stage} {'subj-only' if cond=='subj_only' else 'session-cond.'}")
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=7)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.set_title(title)


def make_figure(traj, out_png):
    fig = plt.figure(figsize=(9, 3.6)); gs = fig.add_gridspec(1, 2, wspace=0.42)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- a: per-session parameter recovery ----
    _bars4(axA, traj, None, "per-session recovery R\u00b2", "Per-session parameter recovery")
    axA.legend(frameon=False, loc="upper center", fontsize=5.5, ncol=2,
               bbox_to_anchor=(0.5, -0.22), columnspacing=1.2, handletextpad=0.4, handlelength=1.2)

    # ---- b: session-position recovery ----
    combos = [("S2", "conditioned"), ("S2", "subj_only"), ("S2b", "conditioned"), ("S2b", "subj_only")]
    labs = ["S2\ncond.", "S2\nsubj-only", "S2b\ncond.", "S2b\nsubj-only"]
    for x, (stage, cond) in enumerate(combos):
        v = traj[(traj.stage == stage) & (traj.cond == cond)].sessfrac_R2.iloc[0]
        axB.bar(x, v, color=BLUE[cond], alpha=ALPHA[stage])
        axB.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(range(4)); axB.set_xticklabels(labs, fontsize=6.5)
    axB.set_ylabel("R\u00b2 recovering session position"); axB.set_ylim(0, 1.05)
    axB.set_title("Session-position recovery")

    for ax in (axA, axB):
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, l in zip((axA, axB), "ab"):
        ax.text(-0.18, 1.04, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-csv", default="stage2b_trajectory_recovery.csv")
    ap.add_argument("--out", default="stage2b_session_trajectory.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.traj_csv), a.out)
    print("wrote", a.out)
