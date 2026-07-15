#!/usr/bin/env python
"""Stage-4b per-session family-switching recovery: subject embedding recovers the stable
Dirichlet mixture weights, but session-conditioning adds ~nothing for decoding the
realized per-session family -- recovery lives at the SUBJECT level, not the session level.

OFFLINE producer (matches the stage1/2/2b/3/4a 'freeze the numbers, key by immutable ID'
convention). Reads two committed, frozen inputs -- NO W&B / network call on this path:
  * stage4b_recovery_grid.csv  -- per-run summary (rel_LL, mix_R2_mean, per-session family
    accuracy session-cond vs subject-only), 5 GRU runs from sweep nptb5bam. The run IDs are
    the human-readable provenance key; the committed CSV is the authoritative frozen copy.
  * stage4b_recovery.json      -- the in-container recovery output (utils.multisubject +
    HierarchicalCognitiveAgents), keyed by run id. Supplies panel c's confusion matrix and
    the per-family mixture-weight R2 breakdown that the flat grid CSV does not carry.

Panels (unchanged from the original committed figure, now reproducible):
  a  mixture-weight recovery R2 vs subject embedding size, session_encoding none (circles)
     vs scalar (squares). Rises steeply with D -- the mixture identity is a high-dimensional
     target that small embeddings cannot hold. Inset: per-family R2 for the scalar D16 run.
  b  per-session family-decoding accuracy vs embedding size, subject-only embedding (grey)
     vs session-conditioned (purple). The two are on top of each other at every D -- the
     session delta adds essentially nothing over the broadcast subject embedding for
     decoding which family was realized in a given session (chance = 1/3).
  c  per-session family confusion matrix (row-normalized) for the scalar D16 run -- modest
     diagonal (0.58-0.65), consistent with panel b's ~0.62 overall.

House convention: session-blind/subject-only = grey, session-conditioned/scalar = purple;
square despined panels, frameless legends. Single seed per cell -> no error bars.
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
})

GREY = "#9e9e9e"
PURPLE = "#7b3294"
FAM_SHORT = {"QLearning": "QLearning", "CompareToThreshold": "CompThr", "LossCounting": "LossCounting"}


def _panel_a(ax, grid, det, cond_run):
    for enc, marker, fill in [("none", "o", "white"), ("scalar", "s", "black")]:
        sub = grid[grid.enc == enc].sort_values("embed")
        ax.plot(sub.embed, sub.mix_R2_mean, "-", color="0.3", lw=1.2, zorder=1)
        ax.scatter(sub.embed, sub.mix_R2_mean, marker=marker,
                   facecolor=fill, edgecolor="black", s=90, zorder=3,
                   label=enc)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16]); ax.set_xticklabels([4, 8, 16])
    ax.set_xlabel("subject embedding size")
    ax.set_ylabel("mixture-weight recovery (R\u00b2)")
    ax.set_title("Mixture-weight recovery\nneeds large embeddings", fontsize=13, loc="left")
    ax.legend(frameon=False, title="session enc.", fontsize=11, title_fontsize=11, loc="upper left")
    pf = det[cond_run]["mix_r2_per_family"]
    txt = "per-family (scalar D%d):\n" % det[cond_run]["emb"] + "\n".join(
        f"  {FAM_SHORT[k.replace('mixweight_','')]}: {v:.2f}" for k, v in pf.items())
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5)


def _panel_b(ax, grid):
    sc = grid[grid.enc == "scalar"].sort_values("embed")
    x = np.arange(len(sc)); w = 0.38
    ax.bar(x - w/2, sc.persession_acc_subjectonly, w, color=GREY, label="subject-only embedding")
    ax.bar(x + w/2, sc.persession_acc_sessioncond, w, color=PURPLE, label="session-conditioned")
    ax.axhline(1/3, ls=":", color="0.6", lw=1.2)
    ax.text(x[-1] + 0.5, 1/3, "chance", va="center", ha="right", fontsize=10, color="0.5")
    ax.set_xticks(x); ax.set_xticklabels([f"D{int(e)}" for e in sc.embed])
    ax.set_xlabel("subject embedding size")
    ax.set_ylabel("per-session family accuracy")
    ax.set_title("Session conditioning adds nothing\nover subject identity", fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=11, loc="upper left")


def _panel_c(ax, det, cond_run):
    rec = det[cond_run]
    cm = np.array(rec["persession_confusion_sessioncond"], float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    labs = [FAM_SHORT[l] for l in rec["persession_labels"]]
    im = ax.imshow(cmn, cmap="Purples", vmin=0, vmax=1)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=14,
                     color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
    ax.set_xlabel("predicted family"); ax.set_ylabel("true family")
    acc = rec["persession_family_acc_sessioncond"]
    ax.set_title(f"Per-session family, {acc:.2f} overall\n(scalar D{rec['emb']}, chance 0.33)",
                 fontsize=13, loc="left")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row-normalized", fontsize=11)


def make_figure(grid, det, cond_run, out_png):
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8))
    _panel_a(axA, grid, det, cond_run)
    _panel_b(axB, grid)
    _panel_c(axC, det, cond_run)
    for ax in (axA, axB):
        for s in ax.spines.values():
            s.set_visible(False)
    axC.set_box_aspect(1)
    for ax, l in zip((axA, axB, axC), "abc"):
        ax.text(-0.16, 1.10, l, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    fig.suptitle("Stage-4b per-session family switching: recovery lives at the SUBJECT level, "
                 "not the session level (Dirichlet(0.5) mixtures, N=200)", fontsize=15, y=1.04)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.16, wspace=0.42)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-csv", default="stage4b_recovery_grid.csv")
    ap.add_argument("--recovery-json", default="stage4b_recovery.json")
    ap.add_argument("--cond-run", default="07y7p2cu", help="scalar D16 run for panels a-inset/c")
    ap.add_argument("--out", default="stage4b_recovery.png")
    a = ap.parse_args()
    grid = pd.read_csv(a.grid_csv)
    det = json.load(open(a.recovery_json))
    make_figure(grid, det, a.cond_run, a.out)
    print("wrote", a.out)
