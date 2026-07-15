#!/usr/bin/env python
"""Stage-4a per-session parameter recovery: baseline_rl vs GRU session-blind vs GRU
session-conditioned, within each true model FAMILY (QLearning / CompareToThreshold /
LossCounting). House-styled to match stage3_recovery_vs_baseline.py's panels b/c
(this figure omits panel a -- fit-quality/relative-likelihood comparison already lives
in stage4a_recovery_combined.png panel a's headline and is not duplicated here; the
NEW content is the per-session recovery panels that stage4a_recovery_combined.png does
not cover).

Unlike Stage 3 (which has NO fixed baseline for RescorlaWagner), all three Stage-4a
families have a CORRECTLY-SPECIFIED fixed baseline fitter (that is the point of Stage 4a
vs Stage 3's missing-fitter gap) -- so all three families get a baseline_rl bar.

  a  Per-session recovery, mean over each family's own DRIFTING parameter set (only
     params carrying a `drift` block in hierarchical_rl_stage4a.yaml have per-session
     ground truth to recover -- static params like forget_rate_unchosen/
     loss_count_threshold_std are excluded, same convention as stage3). CTT's static
     `threshold` is ALSO excluded from this panel's mean (it has no drift block, so
     "per-session recovery" for it is really a subject-level check) but IS scored in
     panel b, per user request -- see panel b caption.
  b  Per-session recovery, per PARAMETER, grouped by family -- the fine-grained view.
     CompareToThreshold's softmax_inverse_temperature and learn_rate are already weakly
     identified at the STATIC (session-blind) level (R2 -0.04 and -1.96 respectively) --
     not a session-conditioning artifact, since it shows up before any per-session
     complication (see stage4a_gru_details.json within-family R2 for run 1xeoeclu). A
     bias/inverse-temperature confound was checked directly against the baseline_rl fits
     and ruled out (corr(biasL, softmax)=0.13 across 200 CTT-baseline-fit subjects); the
     CTT agent's lack of a choice-kernel term likely leaves the likelihood surface poorly
     constrained for softmax_inverse_temperature (fitted values reach ~52 against a true
     ceiling of 15). CTT's STATIC threshold (no drift block, so its bars are a
     subject-level recovery check, not a drift-tracking one) is included in this panel
     per user request: despite 60.5% of its 200 baseline_rl point estimates landing
     outside the true generative range [0.2, 0.6] (down to -1.0), its R2 is actually
     GOOD (baseline 0.85, session-blind 0.79, session-cond. 0.83) -- R2 rewards
     preserved relative ordering/scale across subjects, which survives even when
     individual point estimates are biased or occasionally degenerate. So threshold's
     out-of-range point estimates and its high recovery R2 are not in tension. Together
     with softmax_inverse_temperature/learn_rate's genuinely weak R2, this points to the
     CTT agent class itself (choice_kernel="none", i.e. no history term) giving the DE
     optimizer a flatter, less-constrained likelihood surface than QLearning/LossCounting
     have, rather than anything specific to the session-conditioning procedure. Session-conditioning makes biasL and softmax_inverse_temperature
     recovery WORSE, not merely unhelpful (0.72->0.25, -0.04->-1.85) -- plausibly
     overfitting noise into a cross-validated per-session regression target with little
     real signal to find -- but that mechanism is NOT directly verified here (would need
     the predicted-vs-true scatter for those two cells) and is flagged as an open
     question, not a settled explanation. learn_rate, by contrast, DOES improve with
     conditioning (0.27->0.66), so this is isolated to the two params that were already
     unidentifiable, not a blanket effect on the family.

House convention: black square = baseline_rl, light blue circle = GRU session-blind
(none, D4), dark blue circle = GRU session-conditioned (scalar, D4). Square despined
panels, no legend box.
Offline: reads committed stage4a_persession_recovery.csv.
Single seed (42) per cell -> no error bars.
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
})

COL = {"baseline_rl": "black", "gru_session_blind": "#6baed6", "gru_session_conditioned": "#08519c"}
LAB = {"baseline_rl": "baseline_rl", "gru_session_blind": "GRU (session-blind)",
       "gru_session_conditioned": "GRU (session-cond.)"}
FAM_SHORT = {"QLearning": "QL", "CompareToThreshold": "CTT", "LossCounting": "LC"}


def _panel_a(ax, ps):
    fams = ["QLearning", "CompareToThreshold", "LossCounting"]
    conds = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]
    STATIC_PARAMS = {"threshold"}  # no drift block -- excluded from the drift-recovery mean
    xpos = np.arange(len(fams)); w = 0.27
    for i, cond in enumerate(conds):
        vals = []
        for fam in fams:
            sub = ps[(ps.true_family == fam) & (ps.cond == cond) & (~ps.param.isin(STATIC_PARAMS))]
            vals.append(sub.r2.mean() if len(sub) else np.nan)
        ax.bar(xpos + (i - 1) * w, vals, w, color=COL[cond], label=LAB[cond])
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels([FAM_SHORT[f] for f in fams])
    ax.set_ylabel("mean per-session recovery R\u00b2")
    ax.set_title("Per-session recovery,\nmean over family's drifting params", fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="lower center", bbox_to_anchor=(0.5, -0.40),
              ncol=1, handletextpad=0.4)


def _panel_b(ax, ps):
    fams = ["QLearning", "CompareToThreshold", "LossCounting"]
    conds = ["baseline_rl", "gru_session_blind", "gru_session_conditioned"]
    rows = []
    for fam in fams:
        params = sorted(ps[ps.true_family == fam].param.unique())
        for p in params:
            rows.append((fam, p))
    y = np.arange(len(rows))[::-1]
    for i, cond in enumerate(conds):
        vals = []
        for fam, p in rows:
            sub = ps[(ps.true_family == fam) & (ps.cond == cond) & (ps.param == p)]
            vals.append(sub.r2.iloc[0] if len(sub) else np.nan)
        ax.barh(y - (i - 1) * 0.27, vals, 0.27, color=COL[cond], label=LAB[cond])
    ax.axvline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_yticks(y)
    p_short = {"softmax_inverse_temperature": "softmax_temp", "loss_count_threshold_mean": "loss_thresh_mean"}
    ax.set_yticklabels([f"{FAM_SHORT[f]}: {p_short.get(p, p)}" for f, p in rows], fontsize=11.5)
    ax.set_xlabel("per-session recovery R\u00b2")
    ax.set_title("Per-session recovery, per parameter", fontsize=13, loc="left")
    ax.set_xlim(-2.1, 1.05)


def make_figure(ps, out_png):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.7]})
    _panel_a(axA, ps)
    _panel_b(axB, ps)
    axA.set_box_aspect(1)
    for ax in (axA, axB):
        for s in ax.spines.values():
            s.set_visible(False)
    for ax, l in zip((axA, axB), "ab"):
        ax.text(-0.16, 1.08, l, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    fig.suptitle("Stage 4a \u2014 mixture of model families: per-session parameter "
                 "recovery, baseline vs GRU", fontsize=16, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.24, wspace=0.85)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persession-csv", default="stage4a_persession_recovery.csv")
    ap.add_argument("--out", default="stage4a_persession_recovery.png")
    a = ap.parse_args()
    make_figure(pd.read_csv(a.persession_csv), a.out)
    print("wrote", a.out)
