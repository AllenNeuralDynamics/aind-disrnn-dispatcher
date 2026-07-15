#!/usr/bin/env python
"""Stage-3 fit-quality + per-session parameter recovery, baseline_rl vs GRU, house-styled
to match stage1_recovery_vs_baseline.png / stage2_recovery_vs_baseline.png.

FILLS TWO GAPS that existed for Stage 3 (unlike stages 1/2/2b, which both had a
baseline-vs-GRU likelihood comparison and per-session parameter recovery):

  a  Fit-quality: relative held-out likelihood, baseline_rl (best-of-3 model-selection,
     the realistic deployed scenario; "matched" ceiling if the true model were known is
     shown as a lighter marker) vs the 6 GRU cells (session_encoding x embedding size).
     baseline_rl trails GRU (0.96 best-of-3 / 0.96 matched vs 0.98-0.99 GRU) even before
     accounting for its inability to recover model IDENTITY (see confusion figure).
  b  Per-session parameter recovery, mean over each family's own parameter set, THREE
     conditions per family (baseline_rl / GRU session-blind / GRU session-conditioned) --
     same per-session convention as stages 2/2b (baseline_rl and session-blind broadcast a
     fixed per-subject estimate to every session; only session-conditioned GRU predicts a
     genuinely per-session value). RescorlaWagner has NO baseline_rl bar (no matching fixed
     fitter exists for this stage's toolkit -- see confusion figure panel b).
  c  Per-session recovery, per PARAMETER, grouped by family -- the fine-grained view.
     Several baseline parameters (forget_rate_unchosen, learn_rate, learn_rate_rew,
     choice_kernel_relative_weight) show WEAK identifiability even after winsorizing
     near-degenerate MLE fits at the true parameter ceiling (robust R2 still <=0, though
     Spearman rank correlation 0.22-0.54 confirms partial signal) -- consistent with this
     family's own within-subject session-mean R2 (forget_rate_unchosen was already -0.12
     there). GRU recovers every parameter session-conditioned > session-blind > baseline.

House convention: black square = baseline_rl, light blue circle = GRU session-blind
(none), dark blue circle = GRU session-conditioned (scalar). Square despined panels.
Offline: reads committed stage3_gru_recovery.csv, stage3_baseline_likelihood_summary.csv,
s3_baseline_likelihood.csv, stage3_gru_gt_likelihood.csv, stage3_persession_recovery.csv.
Single seed (42) per cell -> no error bars.
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"baseline_rl": "black", "gru_session_blind": "#6baed6", "gru_session_conditioned": "#08519c"}
MK = {"baseline_rl": "s", "gru_session_blind": "o", "gru_session_conditioned": "o"}
LAB = {"baseline_rl": "baseline_rl", "gru_session_blind": "GRU (session-blind)",
       "gru_session_conditioned": "GRU (session-cond.)"}
FAM_SHORT = {"Bari2019": "Bari", "Hattori2019": "Hatt.", "RescorlaWagner": "RW"}


def _panel_a(ax, gru, base_summary):
    for _, r in gru.iterrows():
        mk = "o" if r.enc == "scalar" else "^"
        col = "#08519c" if r.enc == "scalar" else "#6baed6"
        ax.scatter(r.embed, r.lik_rel, marker=mk, color=col, s=70, edgecolor="white",
                    lw=0.8, zorder=3)
    b3 = base_summary[base_summary.cond == "baseline_rl_best_of_3"].lik_rel.iloc[0]
    bm = base_summary[base_summary.cond == "baseline_rl_matched"].lik_rel.iloc[0]
    ax.axhline(b3, color="black", ls="-", lw=1.6, zorder=1)
    ax.axhline(bm, color="0.5", ls="--", lw=1.1, zorder=1)
    ax.set_xticks([4, 8, 16]); ax.set_xlim(3, 18)
    ax.set_ylim(min(gru.lik_rel.min(), bm) - 0.006, 1.002)
    ax.set_xlabel("subject embedding size"); ax.set_ylabel("relative likelihood")
    ax.set_title(f"Fit quality: GRU 0.98\u20130.99 vs\nbaseline best-of-3={b3:.3f}", fontsize=9, loc="left")
    handles = [plt.Line2D([], [], marker="^", color="#6baed6", lw=0, label="GRU (none)"),
               plt.Line2D([], [], marker="o", color="#08519c", lw=0, label="GRU (scalar)"),
               plt.Line2D([], [], color="black", lw=1.6, label="baseline (best-of-3)"),
               plt.Line2D([], [], color="0.5", lw=1.1, ls="--", label="baseline (matched)")]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.34), ncol=2, handletextpad=0.4)


def _panel_b(ax, ps):
    fams = ["Bari2019", "Hattori2019", "RescorlaWagner"]
    conds = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]
    xpos = np.arange(len(fams)); w = 0.27
    for i, cond in enumerate(conds):
        vals = []
        for fam in fams:
            sub = ps[(ps.true_preset == fam) & (ps.cond == cond)]
            vals.append(sub.r2.mean() if len(sub) else np.nan)
        ax.bar(xpos + (i - 1) * w, vals, w, color=COL[cond], label=LAB[cond])
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels([FAM_SHORT[f] for f in fams])
    ax.set_ylabel("mean per-session recovery R\u00b2")
    ax.set_title("Per-session recovery,\nmean over family's params", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.34),
               ncol=1, handletextpad=0.4)


def _panel_c(ax, ps):
    fams = ["Bari2019", "Hattori2019", "RescorlaWagner"]
    conds = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]
    rows = []
    for fam in fams:
        params = sorted(ps[ps.true_preset == fam].param.unique())
        for p in params:
            rows.append((fam, p))
    y = np.arange(len(rows))[::-1]
    for i, cond in enumerate(conds):
        vals = []
        for fam, p in rows:
            sub = ps[(ps.true_preset == fam) & (ps.cond == cond) & (ps.param == p)]
            vals.append(sub.r2.iloc[0] if len(sub) else np.nan)
        ax.barh(y - (i - 1) * 0.27, vals, 0.27, color=COL[cond], label=LAB[cond])
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{FAM_SHORT[f]}: {p}" for f, p in rows], fontsize=7)
    ax.set_xlabel("per-session recovery R\u00b2")
    ax.set_title("Per-session recovery, per parameter", fontsize=9, loc="left")
    ax.set_xlim(-2.1, 1.05)


def make_figure(gru, base_summary, ps, out_png):
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.6, top=0.80, bottom=0.28)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    _panel_a(axA, gru, base_summary)
    _panel_b(axB, ps)
    _panel_c(axC, ps)
    for ax in (axA, axB):
        ax.set_box_aspect(1)
    for ax in (axA, axB, axC):
        for s in ax.spines.values():
            s.set_visible(False)
    for ax, l in zip((axA, axB, axC), "abc"):
        ax.text(-0.16, 1.10, l, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.suptitle("Stage 3 \u2014 mixture of Q-learning variants: fit quality and per-session "
                 "parameter recovery, baseline vs GRU", fontsize=12, y=0.98)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-csv", default="stage3_gru_recovery.csv")
    ap.add_argument("--baseline-summary-csv", default="stage3_baseline_likelihood_summary.csv")
    ap.add_argument("--persession-csv", default="stage3_persession_recovery.csv")
    ap.add_argument("--out", default="stage3_recovery_vs_baseline.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.gru_csv), pd.read_csv(a.baseline_summary_csv),
                pd.read_csv(a.persession_csv), a.out)
    print("wrote", a.out)
