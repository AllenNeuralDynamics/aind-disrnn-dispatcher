#!/usr/bin/env python
"""Stage-2 recovery figure (3 square panels, house style) — offline producer.

Combines the former stage2_likelihood_comparison + stage2_recovery into one figure
matching stage1_recovery_vs_baseline:
  a  Fit quality: likelihood relative to ground truth vs #subjects
     (baseline_rl, GRU session-blind [none], GRU session-conditioned [scalar]).
  b  Mean subject-parameter recovery R2 vs #subjects (none vs scalar [+ baseline]).
  c  Per-parameter recovery R2 at n=200 (baseline [+] / none / scalar).

Inputs (committed, curated; each keyed by wandb_run_id per the 'freeze the numbers' rule):
  stage2_likelihood_comparison.csv   N, GRU_none, GRU_scalar, baseline_rl  (relative LL)
  stage2_gru_recovery.csv            per (N, enc) subject-param recovery R2 + run id
  stage2_baseline_recovery.csv       OPTIONAL: per-N baseline_rl fitted-param recovery R2
                                     (num_subjects, r2_biasL, r2_learn_rate, r2_softmax_temp).
                                     When absent, the baseline bar/line is omitted and the
                                     figure notes it — baseline's fit-quality story still
                                     shows in panel a. Single seed (42) per cell -> no error bars.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

PARAMS = ["biasL", "learn_rate", "softmax_temp"]
# House color convention (shared with stage 1): light blue = 4-d embedding subject-only
# (session-blind), darker blue = 4-d embedding + session conditioning, black = baseline.
COL = {"none": "#6baed6", "scalar": "#08519c", "base": "black"}


def make_figure(lik, gru, baseline, out_png, focus_n=200):
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

    # ---- b: mean recovery R2 vs N ----
    for enc, mk in [("none", "o"), ("scalar", "o")]:
        g = gru[gru.enc == enc].sort_values("num_subjects")
        axB.plot(g.num_subjects, g.R2_mean, marker=mk, color=COL[enc],
                 label=f"GRU 4-d, {'session-blind' if enc=='none' else '+ session cond.'}")
    if baseline is not None and "r2_mean" in baseline:
        b = baseline.sort_values("num_subjects")
        axB.plot(b.num_subjects, b.r2_mean, marker="s", color=COL["base"], label="baseline_rl")
    axB.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axB.set_xlabel("# subjects"); axB.set_ylabel("mean recovery R\u00b2")
    axB.set_xticks([50, 100, 200, 300]); axB.set_ylim(0.6, 1.03)
    axB.set_title("Parameter recovery"); axB.legend(frameon=False, loc="lower right", fontsize=6)

    # ---- c: per-parameter at N=focus_n ----
    xpos = np.arange(3)
    gn = gru[(gru.enc == "none") & (gru.num_subjects == focus_n)].iloc[0]
    gs2 = gru[(gru.enc == "scalar") & (gru.num_subjects == focus_n)].iloc[0]
    have_base = baseline is not None and (baseline.num_subjects == focus_n).any()
    if have_base:
        b200 = baseline[baseline.num_subjects == focus_n].iloc[0]
        w = 0.27
        axC.bar(xpos - w, [b200[f"r2_{p}"] for p in PARAMS], w, color=COL["base"], label="baseline_rl")
        axC.bar(xpos, [gn[f"r2_{p}"] for p in PARAMS], w, color=COL["none"], label="GRU 4-d, session-blind")
        axC.bar(xpos + w, [gs2[f"r2_{p}"] for p in PARAMS], w, color=COL["scalar"], label="GRU 4-d + session cond.")
    else:
        w = 0.38
        axC.bar(xpos - w / 2, [gn[f"r2_{p}"] for p in PARAMS], w, color=COL["none"], label="GRU 4-d, session-blind")
        axC.bar(xpos + w / 2, [gs2[f"r2_{p}"] for p in PARAMS], w, color=COL["scalar"], label="GRU 4-d + session cond.")
    axC.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axC.set_xticks(xpos); axC.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=7)
    axC.set_ylabel("recovery R\u00b2"); axC.set_ylim(0, 1.05)
    axC.set_title(f"Per-parameter (n={focus_n})")

    for ax in (axA, axB, axC):
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, l in zip((axA, axB, axC), "abc"):
        ax.text(-0.12, 1.02, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lik-csv", default="stage2_likelihood_comparison.csv")
    ap.add_argument("--gru-csv", default="stage2_gru_recovery.csv")
    ap.add_argument("--baseline-csv", default="stage2_baseline_recovery.csv")
    ap.add_argument("--out", default="stage2_recovery_vs_baseline.png")
    a = ap.parse_args()
    lik = pd.read_csv(a.lik_csv); gru = pd.read_csv(a.gru_csv)
    baseline = pd.read_csv(a.baseline_csv) if os.path.exists(a.baseline_csv) else None
    make_figure(lik, gru, baseline, a.out)
    print("wrote", a.out, "(baseline bar:", baseline is not None, ")")
