#!/usr/bin/env python3
"""Kevin-style penalty-selection plot, built from EXISTING study 03 + 05 data (no new compute).

Kevin's "Example Rat / Different DisRNN Penalties" plot is a split-half reliability diagnostic:
score each model on two halves of the data, plot them against each other, colour by penalty, and
pick the penalty that is "upper enough but not separated too far" from the y=x diagonal (high
likelihood AND consistent across halves = not overfit).

ADAPTATION FOR OUR MULTI-SUBJECT SETTING. Kevin's axes are odd/even SESSIONS of one rat
(within-subject reliability). Our models are population models, so we use the population-transfer
analog:
    x = in-sample NormLik   (eval on the TRAINING mice)
    y = held-out NormLik     (eval on UNSEEN mice, the fixed held-out cohort)
Same geometry: on the diagonal = no generalization gap; BELOW the diagonal = the penalty is too
weak and the model overfits the training cohort (fits train mice well, transfers poorly). This is
the more relevant overfitting risk for a foundation model than within-subject odd/even.

DATA (already committed, no W&B needed — regenerates OFFLINE):
  * D=100 : analysis/data/d100_selection.csv  (lr=1e-3 slice of study 03, produced once by
            pull_selection_data.py; the raw study-03 grid has a mislabeled lr column, see that script)
  * D=614 : studies/05-disrnn-scaling-law variant mult-beta-d614 (eval_ll / heldout_ll, beta x mult)

OUTPUT: analysis/fig_model_selection.png  +  analysis/model_selection.json (per-(D,beta) summary).
Colour = base beta; marker = multiplier. The point is to pick beta, THEN fix it and scan mult/other.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent                                # studies/06-disrnn-operating-point-at-scale
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "studies" / "util"))  # shared studies/util
from _meta import build_meta  # noqa: E402

D100_CSV = HERE / "data" / "d100_selection.csv"   # committed lr=1e-3 slice (pull_selection_data.py)
D614_CSV = REPO / "studies/05-disrnn-scaling-law/analysis/grid.csv"
REPORT = HERE / "reports" / "r1-penalty-selection.md"
WANDB_GROUPS = ["updnet-ratio-100mice@20260703-200122", "mult-beta-d614@20260713-003501"]

# discrete colour per base beta (low beta = weak penalty = warm, matching Kevin's low->high ramp)
BETA_COLOR = {0.0003: "#1f77b4", 0.001: "#d62728", 0.003: "#9467bd"}
MULT_MARKER = {1: "o", 2: "s", 5: "^", 10: "x"}


def load_d100() -> pd.DataFrame:
    # committed lr=1e-3 slice; run `make pull` (pull_selection_data.py) to refresh from W&B
    return pd.read_csv(D100_CSV)[["D", "beta", "mult", "seed", "in_ll", "heldout_ll"]]


def load_d614() -> pd.DataFrame:
    g = pd.read_csv(D614_CSV)
    g = g[(g["variant"] == "mult-beta-d614") & (g["state"] == "finished")]
    g = g.rename(columns={"eval_ll": "in_ll", "heldout_ll": "heldout_ll"})
    g = g.dropna(subset=["in_ll", "heldout_ll"])
    g["D"] = 614
    if "seed" not in g:
        g["seed"] = 42
    return g[["D", "beta", "mult", "seed", "in_ll", "heldout_ll"]]


def panel(ax, df: pd.DataFrame, title: str, lo: float, hi: float) -> None:
    ax.plot([lo, hi], [lo, hi], ls="--", c="0.6", lw=1, zorder=0)  # y=x: no generalization gap
    for beta, gb in df.groupby("beta"):
        for mult, gm in gb.groupby("mult"):
            ax.scatter(gm["in_ll"], gm["heldout_ll"],
                       c=BETA_COLOR.get(round(beta, 4), "k"),
                       marker=MULT_MARKER.get(int(mult), "*"),
                       s=70, alpha=0.85, linewidths=1.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("in-sample NormLik  (training mice)")
    ax.set_ylabel("held-out NormLik  (unseen mice)")
    ax.set_title(title)
    ax.grid(alpha=0.2)


def summarize(df: pd.DataFrame) -> list[dict]:
    """Per-(D, beta) centroid + generalization gap (in - heldout). Higher heldout & smaller gap = better."""
    rows = []
    for (D, beta), g in df.groupby(["D", "beta"]):
        rows.append(dict(D=int(D), beta=float(beta),
                         heldout_ll=round(g["heldout_ll"].mean(), 4),
                         in_ll=round(g["in_ll"].mean(), 4),
                         gap=round((g["in_ll"] - g["heldout_ll"]).mean(), 4),
                         n=int(len(g))))
    return sorted(rows, key=lambda r: (r["D"], r["beta"]))


def main() -> None:
    d100, d614 = load_d100(), load_d614()
    alldf = pd.concat([d100, d614], ignore_index=True)

    lo = min(alldf["in_ll"].min(), alldf["heldout_ll"].min()) - 0.002   # shared limits: the
    hi = max(alldf["in_ll"].max(), alldf["heldout_ll"].max()) + 0.002   # gap is comparable across D
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    panel(axes[0], d100, "D=100  (study 03)", lo, hi)
    panel(axes[1], d614, "D=614  (study 05 · mult-beta-d614)", lo, hi)

    beta_handles = [plt.Line2D([], [], marker="o", ls="", c=BETA_COLOR[b], label=f"β={b:g}")
                    for b in sorted(BETA_COLOR)]
    mult_handles = [plt.Line2D([], [], marker=MULT_MARKER[m], ls="", c="0.3", label=f"mult={m}")
                    for m in sorted(MULT_MARKER)]
    axes[0].legend(handles=beta_handles, title="base β (colour)", loc="upper left", fontsize=8)
    axes[1].legend(handles=mult_handles, title="multiplier (marker)", loc="upper left", fontsize=8)
    fig.suptitle("disRNN penalty selection — in-sample vs held-out NormLik "
                 "(population analog of Kevin's odd/even plot)\n"
                 "below the diagonal = overfits the training cohort; pick β high AND near diagonal",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = HERE / "fig_model_selection.png"
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")

    summary = summarize(alldf)
    payload = {"_meta": build_meta("analysis/model_selection.py", WANDB_GROUPS, study_root=STUDY),
               "note": ("axes = in-sample (training mice) vs held-out (unseen mice) NormLik; the "
                        "population-transfer analog of Kevin's within-subject odd/even plot. "
                        "gap = in - heldout > 0 means the penalty overfits the training cohort. "
                        "Both D panels are lr=1e-3 (study 03's lr=1e-3 slice via pull_selection_data.py)."),
               "per_D_beta_centroid": summary}
    (HERE / "model_selection.json").write_text(json.dumps(payload, indent=2))
    update_report_block(summary)
    print("\nper-(D, β) centroid  [heldout | in-sample | gap=in-heldout | n]")
    for r in summary:
        print(f"  D={r['D']:>3}  β={r['beta']:<7g}  heldout={r['heldout_ll']:.4f}  "
              f"in={r['in_ll']:.4f}  gap={r['gap']:+.4f}  (n={r['n']})")


def update_report_block(summary: list[dict]) -> None:
    """Regenerate the <!-- BEGIN result-1 --> table in r1 from the computed centroids."""
    lines = ["| D | β | held-out | in-sample | gap (in−heldout) | n |",
             "|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['D']} | {r['beta']:g} | {r['heldout_ll']:.4f} | "
                     f"{r['in_ll']:.4f} | {r['gap']:+.4f} | {r['n']} |")
    block = "<!-- BEGIN result-1 -->\n" + "\n".join(lines) + "\n<!-- END result-1 -->"
    text = REPORT.read_text()
    new = re.sub(r"<!-- BEGIN result-1 -->.*?<!-- END result-1 -->", block, text, flags=re.S)
    REPORT.write_text(new)
    print(f"regenerated result-1 block in {REPORT.name}")


if __name__ == "__main__":
    main()
