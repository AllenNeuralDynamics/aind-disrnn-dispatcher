#!/usr/bin/env python
"""Stage-2 recovery figure (3 square panels, house style) — offline producer.

One figure matching stage1_recovery_vs_baseline:
  a  Fit quality: likelihood relative to ground truth vs #subjects
     (baseline_rl, GRU session-blind [none], GRU session-conditioned [scalar]).
  b  PER-SESSION parameter recovery R2 (mean over params) vs #subjects, all three models.
  c  Per-session per-parameter recovery R2 at n=200 (baseline / session-blind / session-cond).

Panels b/c score recovery against each SESSION's true drifting parameter (not the
session-mean). Fixed per-subject estimates (baseline_rl, GRU session-blind) are broadcast to
all of a subject's sessions; only the session-conditioned GRU predicts a per-session value.
All three land moderate-to-high because the per-session parameter is dominated by the subject
centroid — session conditioning adds the drift-tracking edge. (Recovery of the drift POSITION
itself, where session-blind is 0 by construction, is a separate story in
stage2_session_trajectory.png — not the target here.)

Inputs (committed, curated; each keyed by wandb_run_id per the 'freeze the numbers' rule):
  stage2_likelihood_comparison.csv   N, GRU_none, GRU_scalar, baseline_rl  (relative LL)
  stage2_persession_recovery.csv     per (N, model) per-session param recovery R2 + run id;
                                     3 models: baseline_rl, gru_session_blind,
                                     gru_session_conditioned. softmax_temp winsorized@20 with
                                     r2_softmax_temp_raw / softmax_spearman disclosed.
                                     Single seed (42) per cell -> no error bars.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

PARAMS = ["biasL", "learn_rate", "softmax_temp"]
# House color convention (shared with stage 1): light blue = 4-d embedding subject-only
# (session-blind), darker blue = 4-d embedding + session conditioning, black = baseline.
COL = {"none": "#6baed6", "scalar": "#08519c", "base": "black"}


def make_figure(lik, ps, out_png, focus_n=200):
    # Match stage1_recovery_vs_baseline exactly: matplotlib default base, per-element
    # fontsize overrides below (titles 10, legend 6, ticklabels 7, panel letters 12).
    fig = plt.figure(figsize=(12, 3.6)); gs = fig.add_gridspec(1, 3, wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- a: fit quality (relative likelihood) ----
    s = lik.sort_values("N")
    axA.plot(s.N, s.baseline_rl, marker="s", color=COL["base"], label="baseline Q-learning")
    axA.plot(s.N, s.GRU_none, marker="o", color=COL["none"], label="GRU 4-d, session-blind")
    axA.plot(s.N, s.GRU_scalar, marker="o", color=COL["scalar"], label="GRU 4-d + session cond.")
    axA.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axA.set_xlabel("# subjects"); axA.set_ylabel("likelihood relative to ground truth")
    axA.set_xticks([50, 100, 200, 300]); axA.set_title("Fit quality (all \u2248 ceiling)")
    axA.set_ylim(0.96, 1.008)  # shared with stage-1 panel a (same quantity) — see house convention
    axA.legend(frameon=False, loc="center right", fontsize=6)

    # ---- b: PER-SESSION parameter recovery R2 vs N ----
    # Target = each SESSION's true drifting parameter (not the session-mean). Fixed
    # per-subject estimates (baseline, GRU session-blind) are broadcast to all sessions;
    # only the session-conditioned GRU predicts a per-session value. All three are moderate
    # because the per-session parameter is dominated by the subject centroid; session
    # conditioning adds the drift-tracking edge.
    series = [("baseline_rl", "base", "s", "baseline_rl"),
              ("gru_session_blind", "none", "o", "GRU 4-d, session-blind"),
              ("gru_session_conditioned", "scalar", "o", "GRU 4-d + session cond.")]
    for model, ck, mk, lab in series:
        d = ps[ps.model == model].sort_values("num_subjects")
        axB.plot(d.num_subjects, d.r2_mean, marker=mk, color=COL[ck], label=lab)
    axB.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axB.set_xlabel("# subjects"); axB.set_ylabel("per-session recovery R\u00b2")
    axB.set_xticks([50, 100, 200, 300]); axB.set_ylim(0.6, 1.03)
    axB.set_title("Per-session parameter recovery"); axB.legend(frameon=False, loc="lower right", fontsize=6)

    # ---- c: per-session per-parameter at N=focus_n ----
    xpos = np.arange(3)
    bars = [("baseline_rl", "base", "baseline_rl"),
            ("gru_session_blind", "none", "GRU 4-d, session-blind"),
            ("gru_session_conditioned", "scalar", "GRU 4-d + session cond.")]
    w = 0.27
    for i, (model, ck, lab) in enumerate(bars):
        row = ps[(ps.model == model) & (ps.num_subjects == focus_n)].iloc[0]
        axC.bar(xpos + (i - 1) * w, [row[f"r2_{p}"] for p in PARAMS], w, color=COL[ck], label=lab)
    axC.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axC.set_xticks(xpos); axC.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=7)
    axC.set_ylabel("per-session recovery R\u00b2"); axC.set_ylim(0, 1.05)
    axC.set_title(f"Per-session per-parameter (n={focus_n})")

    for ax in (axA, axB, axC):
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, l in zip((axA, axB, axC), "abc"):
        ax.text(-0.18, 1.04, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lik-csv", default="stage2_likelihood_comparison.csv")
    ap.add_argument("--persession-csv", default="stage2_persession_recovery.csv")
    ap.add_argument("--out", default="stage2_recovery_vs_baseline.png")
    a = ap.parse_args()
    lik = pd.read_csv(a.lik_csv); ps = pd.read_csv(a.persession_csv)
    make_figure(lik, ps, a.out)
    print("wrote", a.out, "(models:", sorted(ps.model.unique()), ")")
