#!/usr/bin/env python
"""Stage-3 model-IDENTITY confusion: GRU embedding decoding (a) vs fixed-baseline
model-selection (b), house-styled to match the rest of the ladder.

REPLACES the prior stage3_baseline_vs_gru_confusion.png, whose panel-b data was
originally SYNTHESIZED (np.random.choice reverse-engineered to hit a remembered 47%
accuracy number) rather than read from real fits. This producer reads real per-subject
data throughout -- no fabricated data anywhere in this figure's provenance.

  a  GRU embedding decoding: 3-way logistic regression on the subject embedding
     (5-fold CV), run 1y7vz70o (none, D4) -- 98.5% correct, near-perfect diagonal.
  b  Fixed-baseline model-selection: fit ALL FOUR RL fitters (Bari/Hattori/
     CompareToThreshold/RescorlaWagner) per subject, assign to whichever gives the
     best held-out likelihood -- 51.0% correct (s3_baseline_modelselection_4way.csv).
     RescorlaWagner's own fitter now exists (baseline-rw-stage3 sweep, wandb run
     qy9lof3x) and correctly recovers 8/67 true-RW subjects (0/67 was structurally
     guaranteed under the earlier 3-fitter setup, since no RW option existed at all).
     Most true-RW subjects (35/67) still get misassigned to Bari, and 19/67 to
     Hattori -- so RW's OWN fitter frequently does not win even on its own true
     subjects. This points to RescorlaWagner being weakly identified relative to the
     softmax-QL family under this held-out-likelihood criterion, not merely "missing
     from the toolkit" -- adding the correct fitter closed most but not all of the gap
     (47.0%->51.0%; GRU embedding decoding still wins by a wide margin at 98.5%).
     A 3-fitter version of this analysis (s3_baseline_modelselection.csv, structurally
     0/67 for RW) is retained for historical comparison via --n-baselines 3.

Offline: reads committed CSV/JSON inputs, selectable via --n-baselines {3,4}.
"""
import argparse
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
})


def _cm_panel(ax, cm, row_labels, col_labels, acc, title, cmap):
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            frac = v / max(cm.max(), 1)
            ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                     color="white" if frac > 0.5 else "black")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_xlabel("assigned"); ax.set_ylabel("true model type")
    ax.set_title(f"{title}\n{acc*100:.1f}% correct", fontsize=15, loc="left")


def make_figure(gru_cm, gru_labels, gru_acc, base_cm, base_row_labels, base_col_labels, base_acc, out_png):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 5.0))
    short = {"Bari2019": "Bari", "Hattori2019": "Hatt.", "RescorlaWagner": "RW",
             "CompareToThreshold": "CTT"}
    _cm_panel(axA, gru_cm, [short[l] for l in gru_labels], [short[l] for l in gru_labels],
              gru_acc, "GRU embedding decoding", plt.get_cmap("Blues"))
    _cm_panel(axB, base_cm, [short[l] for l in base_row_labels], [short[l] for l in base_col_labels],
              base_acc, "Fixed-baseline model selection", plt.get_cmap("Oranges"))
    for ax in (axA, axB):
        ax.set_box_aspect(1)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax, l in zip((axA, axB), "ab"):
        ax.text(-0.18, 1.12, l, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    fig.suptitle("Stage 3 \u2014 recovering model identity: the embedding decodes it "
                 f"({gru_acc*100:.1f}%); fixed-baseline selection cannot ({base_acc*100:.1f}%)",
                 fontsize=15, y=1.03)
    fig.subplots_adjust(left=0.09, right=0.95, top=0.84, bottom=0.10, wspace=0.45)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms-csv", default="s3_baseline_modelselection_4way.csv",
                     help="3-fitter file: s3_baseline_modelselection.csv (RW structurally excluded); "
                          "4-fitter file: s3_baseline_modelselection_4way.csv (RW's own fitter included)")
    ap.add_argument("--gru-details-json", default="stage3_gru_details.json")
    ap.add_argument("--gru-run", default="1y7vz70o")
    ap.add_argument("--n-baselines", type=int, choices=[3, 4], default=4,
                     help="3 = historical (Bari/Hattori/CTT only, RW has no matching fitter); "
                          "4 = current (adds RescorlaWagner's own fitter)")
    ap.add_argument("--out", default="stage3_baseline_vs_gru_confusion.png")
    a = ap.parse_args()

    det = json.load(open(a.gru_details_json))
    d = det[a.gru_run]
    gru_cm = np.array(d["cm"]); gru_labels = d["labels"]; gru_acc = d["acc"]

    ms = pd.read_csv(a.ms_csv)
    presets = ["Bari2019", "Hattori2019", "RescorlaWagner"]
    baselines_avail = ["Bari2019", "Hattori2019", "CompareToThreshold"]
    if a.n_baselines == 4:
        baselines_avail = baselines_avail + ["RescorlaWagner"]
    base_cm = pd.crosstab(ms.true_preset, ms.selected_baseline).reindex(
        index=presets, columns=baselines_avail, fill_value=0).values
    base_acc = (ms.true_preset == ms.selected_baseline).mean()

    make_figure(gru_cm, gru_labels, gru_acc, base_cm, presets, baselines_avail, base_acc, a.out)
    print("wrote", a.out)
