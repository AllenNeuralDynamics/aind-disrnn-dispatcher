#!/usr/bin/env python
"""Stage-2b SESSION-TRAJECTORY recovery (panels a,b), house-styled to match stage 1/2.

  a  Per-session PARAMETER recovery R2 at Stage 2b, n=200, THREE models (baseline_rl /
     GRU session-blind / GRU session-conditioned) — the exact stage2_recovery_figure.py
     panel-c format (same colors: black/light-blue/dark-blue, same bar layout), for direct
     side-by-side comparison with the Stage-2 combined figure.
  b  Session-POSITION recovery (sessfrac R2): how well the embedding locates a session
     within its subject, Stage 2 vs Stage 2b, subject-only vs session-conditioned.
     Subject-only is 0.00 by construction (a fixed per-subject point carries no session
     index); session conditioning recovers it (S2 0.94 -> S2b 0.47 as the drift turns
     non-monotonic). baseline_rl has no session-position estimate (fixed per-subject fit
     only) so it is not shown in panel b.

House convention: black = baseline, light blue = session-blind/subject-only, dark blue =
session-conditioned; in panel b, stage is distinguished by saturation (Stage-2 lighter,
Stage-2b full). Square despined panels, presentation-scale fonts. Offline: reads the
committed curated stage2b_trajectory_recovery.csv (frozen; baseline_rl S2b row from
compute_stage2b_baseline_persession.py, same broadcast method as the Stage-2 baseline
recompute). Single seed (42) per cell -> no error bars.

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

# Panel-a house convention identical to stage2_recovery_figure.py panel c: baseline = black,
# session-blind = light blue, session-conditioned = dark blue.
COL = {"baseline_rl": "black", "subj_only": "#6baed6", "conditioned": "#08519c"}
LAB = {"baseline_rl": "baseline_rl", "subj_only": "GRU 4-d, session-blind",
       "conditioned": "GRU 4-d + session cond."}


def _bars_s2b(ax, df, ylabel, title):
    """Stage-2b only, 3 bars/parameter (baseline / session-blind / session-cond) —
    exactly the stage2_recovery_figure.py panel-c format, for direct side-by-side."""
    xpos = np.arange(len(PARAMS)); w = 0.27
    order = ["baseline_rl", "subj_only", "conditioned"]
    for i, cond in enumerate(order):
        row = df[(df.stage == "S2b") & (df.cond == cond)].iloc[0]
        vals = [row[f"r2_{p}"] for p in PARAMS]
        ax.bar(xpos + (i - 1) * w, vals, w, color=COL[cond], label=LAB[cond])
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=7)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.set_title(title)


def make_figure(traj, out_png):
    fig = plt.figure(figsize=(9, 3.6)); gs = fig.add_gridspec(1, 2, wspace=0.42)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- a: per-session parameter recovery, Stage 2b only (matches stage2 panel c) ----
    _bars_s2b(axA, traj, "per-session recovery R\u00b2", "Per-session (Stage 2b, n=200)")
    axA.legend(frameon=False, loc="upper center", fontsize=5.5, ncol=1,
               bbox_to_anchor=(0.5, -0.24), handletextpad=0.4, handlelength=1.2)

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
