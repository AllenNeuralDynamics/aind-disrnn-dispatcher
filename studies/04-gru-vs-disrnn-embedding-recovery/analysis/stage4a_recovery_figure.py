"""Stage 4a combined recovery figure -- family mixture (QLearning / CompareToThreshold /
LossCounting): embedding-space PCA (a,b), GRU embedding decoding confusion (c), fixed-baseline
model-selection confusion (d). All four panels square per house convention.

Freezes the numbers per the posthoc-reporting 'freeze the numbers, key them by an immutable
ID -- never a live handle' rule. Reads four committed curated inputs:
  - stage4a_embedding.csv               (200 subjects x 8-d embedding + true params, run cq650txm)
  - stage4a_gru_details.json            (per-run confusion matrices, keyed by wandb_run_id)
  - stage4a_gru_recovery.csv            (embed-size grid: run/enc/embed/fam_acc/chance)
  - stage4a_baseline_modelselection.csv (200 subjects: true_family, selected_baseline)

Offline, no W&B. Originally generated ad hoc (no committed producer) -- this script and its
frozen inputs close that gap so `make` can regenerate the figure without a live artifact lookup.

Usage:
    python stage4a_recovery_figure.py \
        --emb-csv stage4a_embedding.csv \
        --gru-details-json stage4a_gru_details.json \
        --gru-recovery-csv stage4a_gru_recovery.csv \
        --ms-csv stage4a_baseline_modelselection.csv \
        --out figures/stage4a_recovery_combined.png
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                      "figure.dpi": 110, "axes.titlesize": 15, "axes.labelsize": 14,
                      "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12})

FAMS = ["QLearning", "CompareToThreshold", "LossCounting"]
SHORT = {"QLearning": "QL", "CompareToThreshold": "CTT", "LossCounting": "LossCnt"}
MARKERS = {"QLearning": "o", "CompareToThreshold": "s", "LossCounting": "D"}
FAMCOL = {"QLearning": "#4C72B0", "CompareToThreshold": "#DD8452", "LossCounting": "#55A868"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-csv", required=True)
    ap.add_argument("--gru-details-json", required=True)
    ap.add_argument("--gru-recovery-csv", required=True)
    ap.add_argument("--ms-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    emb = pd.read_csv(args.emb_csv)
    det = json.load(open(args.gru_details_json))
    rec = pd.read_csv(args.gru_recovery_csv)
    ms = pd.read_csv(args.ms_csv)

    X = StandardScaler().fit_transform(emb[[c for c in emb.columns if c.startswith("embedding_")]].values)
    Z = PCA(2, random_state=0).fit_transform(X)
    emb["z1"], emb["z2"] = Z[:, 0], Z[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axa, axb, axc, axd = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) embedding PCA colored by true family
    for f in FAMS:
        s = emb[emb.fam == f]
        axa.scatter(s.z1, s.z2, marker=MARKERS[f], c=FAMCOL[f], s=30, edgecolor="white",
                    linewidth=0.4, label=SHORT[f], alpha=0.9)
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_xlabel("embedding PC1"); axa.set_ylabel("PC2")
    axa.set_title("Embedding separates model families", loc="left")
    axa.legend(fontsize=11.5, frameon=False, loc="best", title="true family", title_fontsize=11.5)
    axa.set_box_aspect(1)

    # (b) embedding PCA colored by true biasL
    vabs = max(abs(emb.param_biasL.min()), abs(emb.param_biasL.max()))
    scb = None
    for f in FAMS:
        s = emb[emb.fam == f]
        scb = axb.scatter(s.z1, s.z2, marker=MARKERS[f], c=s.param_biasL, cmap="coolwarm",
                           vmin=-vabs, vmax=vabs, s=30, edgecolor="#333", linewidth=0.3)
    fig.colorbar(scb, ax=axb, fraction=0.046, pad=0.04).set_label("biasL", fontsize=13)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_xlabel("PC1"); axb.set_ylabel("PC2")
    axb.set_title("Shape = family,  color = true biasL", loc="left")
    hh = [Line2D([0], [0], marker=MARKERS[f], color="none", markerfacecolor="#888",
                 markeredgecolor="#333", markersize=10, label=SHORT[f]) for f in FAMS]
    axb.legend(handles=hh, fontsize=11.5, frameon=False, loc="best")
    axb.set_box_aspect(1)

    # (c) GRU embedding decoding confusion (smallest embed size in the recovery grid)
    run0 = rec.sort_values("embed").iloc[0]["run"]
    gcm = np.array(det[run0]["cm"]); glab = det[run0]["labels"]
    gacc = np.trace(gcm) / gcm.sum()
    cmn = gcm / gcm.sum(1, keepdims=True)
    axc.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axc.set_xticks(range(3)); axc.set_xticklabels([SHORT[l] for l in glab])
    axc.set_yticks(range(3)); axc.set_yticklabels([SHORT[l] for l in glab])
    axc.set_xlabel("predicted (embedding decode)"); axc.set_ylabel("true family")
    axc.set_title(f"GRU embedding decoding \u2014 {gacc*100:.1f}%", loc="left")
    for i in range(3):
        for j in range(3):
            axc.text(j, i, f"{gcm[i, j]}", ha="center", va="center", fontsize=16,
                     color="white" if cmn[i, j] > 0.5 else "#333")

    # (d) fixed-baseline model-selection confusion
    bcm = pd.crosstab(ms.true_family, ms.selected_baseline).reindex(index=FAMS, columns=FAMS,
                                                                     fill_value=0).values
    bacc = (ms.true_family == ms.selected_baseline).mean()
    bcmn = bcm / bcm.sum(1, keepdims=True)
    axd.imshow(bcmn, cmap="Oranges", vmin=0, vmax=1, aspect="equal")
    axd.set_xticks(range(3)); axd.set_xticklabels([SHORT[f] for f in FAMS])
    axd.set_yticks(range(3)); axd.set_yticklabels([SHORT[f] for f in FAMS])
    axd.set_xlabel("selected family (best-fit likelihood)"); axd.set_ylabel("true family")
    axd.set_title(f"Fixed-baseline model selection \u2014 {bacc*100:.1f}%", loc="left")
    for i in range(3):
        for j in range(3):
            axd.text(j, i, f"{bcm[i, j]}", ha="center", va="center", fontsize=16,
                     color="white" if bcmn[i, j] > 0.5 else "#333")
    # QL(row0) -> CTT(col1) confusion cell, highlighted (dominant baseline error mode)
    axd.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor="#C44E52", lw=2.2))

    for ax, L in zip([axa, axb, axc, axd], "abcd"):
        ax.text(-0.08, 1.08, L, transform=ax.transAxes, fontweight="bold", fontsize=19,
                va="top", ha="right")

    fig.suptitle("Stage 4a \u2014 family mixture (QL / CTT / LossCounting):\n"
                 "embedding decodes family at 100% vs 70% fixed-baseline selection",
                 fontsize=14.5, y=0.99)
    fig.subplots_adjust(left=0.09, right=0.90, top=0.84, bottom=0.06, hspace=0.36, wspace=0.34)
    fig.canvas.draw()
    # enforce a strict square DATA box for every panel (post-layout, since tight_layout/
    # subplots_adjust can otherwise leave non-square axes even with set_box_aspect(1) when a
    # colorbar or long tick labels eat into one panel's width but not its neighbor's).
    for ax in [axa, axb, axc, axd]:
        ax.set_box_aspect(1)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
