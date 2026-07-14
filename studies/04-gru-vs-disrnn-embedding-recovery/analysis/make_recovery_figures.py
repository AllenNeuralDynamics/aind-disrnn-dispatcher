#!/usr/bin/env python
"""Generate Stage-1 recovery figures from the scores tables.

Inputs (produced by run_recovery_analysis.py + the baseline parser):
  - GRU recovery scores CSV (one row per finished GRU run: subj,hid,emb,r2_*,cca_*)
  - baseline_rl recovery CSV (num_subjects, r2_learn_rate, r2_biasL, r2_softmax_temp,
    r2_mean, eval_likelihood)
  - ground-truth likelihood per subject count (for baseline relative-likelihood)

Panels:
  A  mean recovery R2 vs #subjects: GRU variants vs correct-model baseline
  B  fit likelihood RELATIVE to ground truth (ceiling 1.0) -- both model classes
     MUST be on the relative scale: baseline logs ABSOLUTE eval_likelihood, so
     baseline_relative = eval_likelihood / groundtruth_likelihood. Plotting
     baseline absolute against GRU relative is a scale error that makes the
     correct-model reference look worse than ceiling -- do not do it.
  C  per-parameter recovery R2 at a chosen N: GRU embed=4 vs baseline
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

PARAMS = ["biasL", "learn_rate", "softmax_temp"]


def make_figure(gru, baseline, gt_by_subj, out_png, focus_n=200, hid_focus=16):
    gcol = {"h16e4": "#C44E52", "h64e4": "#8172B3", "e2": "#CCB974", "base": "#000000"}
    fig = plt.figure(figsize=(12, 3.6)); gs = fig.add_gridspec(1, 3, wspace=0.42)
    # panel order (left->right): B fit quality, A parameter recovery, C per-parameter
    axB = fig.add_subplot(gs[0, 0]); axA = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])

    def line(ax, d, x, y, **kw):
        s = d.sort_values(x); ax.plot(s[x], s[y], marker="o", ms=5, **kw)

    line(axA, gru[(gru.hid == 16) & (gru.emb == 4)], "subj", "r2_r2_mean", color=gcol["h16e4"], label="GRU h16 e4")
    if len(gru[(gru.hid == 64) & (gru.emb == 4)]):
        line(axA, gru[(gru.hid == 64) & (gru.emb == 4)], "subj", "r2_r2_mean", color=gcol["h64e4"], label="GRU h64 e4")
    line(axA, gru[gru.emb == 2].groupby("subj", as_index=False).r2_r2_mean.mean(), "subj", "r2_r2_mean",
         color=gcol["e2"], ls="--", label="GRU e2 (mean)")
    line(axA, baseline.rename(columns={"num_subjects": "subj"}), "subj", "r2_mean",
         color=gcol["base"], lw=2.2, label="baseline_rl (correct model)")
    axA.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axA.set_xlabel("# subjects"); axA.set_ylabel("mean recovery R\u00b2"); axA.set_xticks([50, 100, 200, 300])
    axA.set_ylim(0.2, 1.03); axA.set_title("Parameter recovery", fontsize=10)
    axA.legend(fontsize=6, frameon=False, loc="lower right")

    bl = baseline.copy(); bl["lik_rel"] = bl.apply(lambda r: r.eval_likelihood / gt_by_subj[int(r.num_subjects)], axis=1)
    axB.plot(bl.num_subjects, bl.lik_rel, marker="s", ms=6, color=gcol["base"], lw=2, label="baseline_rl (correct model)")
    for emb, col, ls, lab in [(4, gcol["h16e4"], "-", "GRU h16 e4"), (2, gcol["e2"], "--", "GRU h16 e2")]:
        g = gru[(gru.hid == 16) & (gru.emb == emb)].sort_values("subj")
        if "lik_rel" in g:
            axB.plot(g.subj, g.lik_rel, marker="o", ms=5, color=col, ls=ls, label=lab)
    axB.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axB.set_xlabel("# subjects"); axB.set_ylabel("likelihood relative to ground truth")
    axB.set_xticks([50, 100, 200, 300]); axB.set_ylim(0.96, 1.008)
    axB.set_title("Fit quality (all \u2248 ceiling)", fontsize=10); axB.legend(fontsize=6, frameon=False, loc="lower right")

    xpos = np.arange(3); w = 0.27
    g200 = gru[(gru.hid == hid_focus) & (gru.emb == 4) & (gru.subj == focus_n)].iloc[0]
    g200e2 = gru[(gru.hid == hid_focus) & (gru.emb == 2) & (gru.subj == focus_n)].iloc[0]
    b200 = baseline[baseline.num_subjects == focus_n].iloc[0]
    axC.bar(xpos - w, [b200[f"r2_{p}"] for p in PARAMS], w, color="black", label="baseline_rl")
    axC.bar(xpos, [g200e2[f"r2_{p}"] for p in PARAMS], w, color=gcol["e2"], label=f"GRU h{hid_focus} e2")
    axC.bar(xpos + w, [g200[f"r2_{p}"] for p in PARAMS], w, color=gcol["h16e4"], label=f"GRU h{hid_focus} e4")
    axC.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    axC.set_xticks(xpos); axC.set_xticklabels(PARAMS, fontsize=7); axC.set_ylabel("recovery R\u00b2")
    axC.set_ylim(0, 1.05); axC.set_title(f"Per-parameter (n={focus_n})", fontsize=10)

    for ax in (axA, axB, axC):
        ax.set_box_aspect(1)  # roughly square panels
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, l in zip((axB, axA, axC), "abc"):
        ax.text(-0.12, 1.02, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru-csv", required=True)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--gt-json", required=True, help='JSON {subject_count: groundtruth_likelihood}')
    ap.add_argument("--gru-likrel-json", default=None, help='optional JSON {run_id: likelihood_relative_to_groundtruth}')
    ap.add_argument("--out", default="stage1_recovery_vs_baseline.png")
    a = ap.parse_args()
    gru = pd.read_csv(a.gru_csv); baseline = pd.read_csv(a.baseline_csv)
    gt = {int(k): v for k, v in json.load(open(a.gt_json)).items()}
    if a.gru_likrel_json:
        lr = json.load(open(a.gru_likrel_json)); gru["lik_rel"] = gru["run"].map(lr)
    make_figure(gru, baseline, gt, a.out)
    print("wrote", a.out)
