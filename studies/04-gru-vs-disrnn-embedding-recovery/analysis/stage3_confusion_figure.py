#!/usr/bin/env python
"""Stage-3 model-IDENTITY confusion: GRU embedding decoding (a) vs fixed-baseline
model-selection (b), house-styled to match the rest of the ladder.

REPLACES the prior stage3_baseline_vs_gru_confusion.png, whose panel-b data was
originally SYNTHESIZED (np.random.choice reverse-engineered to hit a remembered 47%
accuracy number) rather than read from real fits. This producer reads real per-subject
data throughout -- no fabricated data anywhere in this figure's provenance.

  a  GRU embedding decoding: 3-way logistic regression on the subject embedding
     (5-fold CV), run 1y7vz70o (none, D4) -- 98.5% correct, near-perfect diagonal.
  b  Fixed-baseline model-selection, MATCHED comparison (default, --baselines matched):
     exactly the 3 true generative fitters (Bari/Hattori/RescorlaWagner) compete per
     subject -- CompareToThreshold, which has no matching true preset in this stage,
     is dropped entirely (s3_baseline_modelselection_3true.csv). 62.5% correct (125/200),
     UP from 51.0% once CTT stops siphoning off Bari/Hattori subjects as a spurious
     best fit. RescorlaWagner's own fitter (baseline-rw-stage3 sweep, wandb run
     qy9lof3x) still only recovers 8/66 true-RW subjects even with no off-target
     competitor in the mix -- 36/66 still misassigned to Bari, 22/66 to Hattori -- so
     RW remains weakly identified relative to the softmax-QL family under this
     held-out-likelihood criterion; this is a genuine property of RW's fit, not an
     artifact of CTT crowding the comparison. GRU embedding decoding still wins by a
     wide margin either way (98.5%). Two other selectable modes for comparison:
     --baselines historical (Bari/Hattori/CTT, no RW fitter at all, 47.0%) and
     --baselines plus_rw (adds RW alongside CTT, 4-way, 51.0%).

Offline: reads committed CSV/JSON inputs, selectable via --baselines {historical,plus_rw,matched}.
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
    ap.add_argument("--ms-csv", default="s3_baseline_modelselection_3true.csv",
                     help="s3_baseline_modelselection.csv (3-fitter historical: Bari/Hattori/CTT, RW "
                          "structurally excluded); s3_baseline_modelselection_4way.csv (4-fitter: adds "
                          "RW's own fitter, off-target CTT still competes); "
                          "s3_baseline_modelselection_3true.csv (3-fitter MATCHED: exactly the 3 true "
                          "generative fitters Bari/Hattori/RW, CTT excluded -- default, apples-to-apples)")
    ap.add_argument("--gru-details-json", default="stage3_gru_details.json")
    ap.add_argument("--gru-run", default="1y7vz70o")
    ap.add_argument("--baselines", default="matched", choices=["historical", "plus_rw", "matched"],
                     help="historical = Bari/Hattori/CTT only (RW has no matching fitter, pre-qy9lof3x); "
                          "plus_rw = adds RescorlaWagner's own fitter, CTT still competes as a 4th "
                          "off-target option; matched = exactly the 3 true generative fitters "
                          "(Bari/Hattori/RW), CTT dropped entirely -- apples-to-apples 3-vs-3 comparison")
    ap.add_argument("--out", default="stage3_baseline_vs_gru_confusion.png")
    a = ap.parse_args()

    det = json.load(open(a.gru_details_json))
    d = det[a.gru_run]
    gru_cm = np.array(d["cm"]); gru_labels = d["labels"]; gru_acc = d["acc"]

    ms = pd.read_csv(a.ms_csv)
    presets = ["Bari2019", "Hattori2019", "RescorlaWagner"]
    baselines_by_mode = {
        "historical": ["Bari2019", "Hattori2019", "CompareToThreshold"],
        "plus_rw": ["Bari2019", "Hattori2019", "CompareToThreshold", "RescorlaWagner"],
        "matched": ["Bari2019", "Hattori2019", "RescorlaWagner"],
    }
    baselines_avail = baselines_by_mode[a.baselines]
    base_cm = pd.crosstab(ms.true_preset, ms.selected_baseline).reindex(
        index=presets, columns=baselines_avail, fill_value=0).values
    base_acc = (ms.true_preset == ms.selected_baseline).mean()

    make_figure(gru_cm, gru_labels, gru_acc, base_cm, presets, baselines_avail, base_acc, a.out)
    print("wrote", a.out)
