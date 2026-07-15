#!/usr/bin/env python
"""Stage-3 model-IDENTITY confusion: GRU embedding decoding (a) vs fixed-baseline
model-selection (b), house-styled to match the rest of the ladder.

REPLACES the prior stage3_baseline_vs_gru_confusion.png, whose panel-b data was
SYNTHESIZED (np.random.choice reverse-engineered to hit a remembered 47% accuracy
number) rather than read from the real per-subject CSV. This producer reads the real
s3_baseline_modelselection.csv (200 real subjects, real per-subject baseline
assignments) and the real stage3_gru_details.json confusion matrix -- no fabricated
data anywhere in this figure's provenance.

  a  GRU embedding decoding: 3-way logistic regression on the subject embedding
     (5-fold CV), run 1y7vz70o (none, D4) -- 98.5% correct, near-perfect diagonal.
  b  Fixed-baseline model-selection: fit ALL THREE available baselines (Bari/
     Hattori/CompareToThreshold) per subject, assign to whichever gives the best
     held-out likelihood -- 47.0% correct. RescorlaWagner has NO matching baseline
     (by design -- only 3 fixed fitters exist for this stage's toolkit) so its
     subjects are structurally mis-assigned to the closest available softmax-QL
     fitter (mostly Bari).

Offline: reads the committed s3_baseline_modelselection.csv and stage3_gru_details.json.
"""
import argparse
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cm_panel(ax, cm, row_labels, col_labels, acc, title, cmap):
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            frac = v / max(cm.max(), 1)
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                     color="white" if frac > 0.5 else "black")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("assigned"); ax.set_ylabel("true model type")
    ax.set_title(f"{title}\n{acc*100:.1f}% correct", fontsize=10, loc="left")


def make_figure(gru_cm, gru_labels, gru_acc, base_cm, base_row_labels, base_col_labels, base_acc, out_png):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 4.2))
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
        ax.text(-0.18, 1.12, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.suptitle("Stage 3 \u2014 recovering model identity: the embedding decodes it "
                 f"({gru_acc*100:.1f}%); fixed-baseline selection cannot ({base_acc*100:.1f}%)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms-csv", default="s3_baseline_modelselection.csv")
    ap.add_argument("--gru-details-json", default="stage3_gru_details.json")
    ap.add_argument("--gru-run", default="1y7vz70o")
    ap.add_argument("--out", default="stage3_baseline_vs_gru_confusion.png")
    a = ap.parse_args()

    det = json.load(open(a.gru_details_json))
    d = det[a.gru_run]
    gru_cm = np.array(d["cm"]); gru_labels = d["labels"]; gru_acc = d["acc"]

    ms = pd.read_csv(a.ms_csv)
    presets = ["Bari2019", "Hattori2019", "RescorlaWagner"]
    baselines_avail = ["Bari2019", "Hattori2019", "CompareToThreshold"]
    base_cm = pd.crosstab(ms.true_preset, ms.selected_baseline).reindex(
        index=presets, columns=baselines_avail, fill_value=0).values
    base_acc = (ms.true_preset == ms.selected_baseline).mean()

    make_figure(gru_cm, gru_labels, gru_acc, base_cm, presets, baselines_avail, base_acc, a.out)
    print("wrote", a.out)
