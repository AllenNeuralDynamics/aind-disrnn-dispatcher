#!/usr/bin/env python
"""Stage-3 GRU session-blind: true INITIAL (session-0, pre-drift) parameter vs GRU's
within-family cross-validated static prediction, per true family (Bari2019/Hattori2019/
RescorlaWagner). Companion to stage3_initial_param_scatter.py (baseline_rl's analog),
using the identical panel layout/format for direct visual comparison.

Prediction target/method: GRU session-blind's subject embedding (run 1y7vz70o) fed
through a within-family GroupKFold + LinearRegression + cross_val_predict against the
session-MEAN of each parameter (same "session-blind" estimator used throughout
compute_stage3_persession_recovery.py), evaluated here against the true SESSION-0 value
specifically (not the session mean) -- matching the baseline scatter's x-axis exactly.

No winsorization ceiling is applied to the GRU predictions: unlike baseline_rl's
per-subject independent MLE, GRU's predictions never blow past the true plausible range
(the shared cross-subject readout acts as implicit shrinkage), so there are no
degenerate-fit outliers to clip.

Inputs: stage3_gru_initial_param_scatter.csv (frozen: true_preset, subject_id, param,
true_initial, gru_pred_static).

Offline: reads committed CSV, no network needed.
"""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_PARAMS = {
    "Bari2019": ["biasL", "choice_kernel_relative_weight", "forget_rate_unchosen",
                 "learn_rate", "softmax_inverse_temperature"],
    "Hattori2019": ["biasL", "learn_rate_rew", "learn_rate_unrew", "softmax_inverse_temperature"],
    "RescorlaWagner": ["biasL", "learn_rate", "epsilon"],
}
FAM_SHORT = {"Bari2019": "Bari", "Hattori2019": "Hattori", "RescorlaWagner": "RescorlaWagner"}
PARAM_LABEL = {
    "biasL": "biasL", "choice_kernel_relative_weight": "choice-kernel wt",
    "forget_rate_unchosen": "forget rate (unchosen)", "learn_rate": "learn rate",
    "softmax_inverse_temperature": "inverse temp.", "learn_rate_rew": "learn rate (rew)",
    "learn_rate_unrew": "learn rate (unrew)", "epsilon": "epsilon",
}


def make_figure(df, out_png):
    nrows = len(FAMILY_PARAMS)
    ncols = max(len(v) for v in FAMILY_PARAMS.values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))

    for i, (fam, params) in enumerate(FAMILY_PARAMS.items()):
        for j in range(ncols):
            ax = axes[i, j]
            if j >= len(params):
                ax.axis("off")
                continue
            p = params[j]
            gp = df[(df.true_preset == fam) & (df.param == p)]
            x = gp.true_initial.values
            y = gp.gru_pred_static.values
            r2 = r2_score(x, y)
            rho = spearmanr(x, y).correlation

            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            pad = 0.06 * (lims[1] - lims[0] if lims[1] > lims[0] else 1)
            lims = [lims[0] - pad, lims[1] + pad]
            ax.plot(lims, lims, color="0.75", lw=1, ls="--", zorder=1)
            ax.scatter(x, y, s=18, color="#08519c", alpha=0.65, edgecolor="white", lw=0.3, zorder=2)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_box_aspect(1)
            ax.locator_params(axis="both", nbins=4)
            ax.set_title(f"{PARAM_LABEL[p]}\nR\u00b2={r2:.2f}, \u03c1={rho:.2f}", fontsize=8.5, loc="left")
            ax.set_xlabel("true (session 0)", fontsize=8)
            ax.set_ylabel("GRU session-blind pred.", fontsize=8, labelpad=2)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i, 0].text(-0.55, 0.5, FAM_SHORT[fam], transform=axes[i, 0].transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)

    fig.suptitle("Stage 3 \u2014 GRU session-blind: true initial (session-0) parameter vs "
                "within-family CV prediction", fontsize=12, y=1.05)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scatter-csv", default="stage3_gru_initial_param_scatter.csv")
    ap.add_argument("--out", default="stage3_gru_initial_param_scatter.png")
    a = ap.parse_args()

    df = pd.read_csv(a.scatter_csv)
    make_figure(df, a.out)
    print("wrote", a.out)
