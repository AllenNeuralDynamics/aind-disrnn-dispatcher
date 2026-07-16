#!/usr/bin/env python
"""Stage-4a GRU session-blind: true INITIAL (session-0, pre-drift) parameter vs GRU's
within-family cross-validated (5-fold GroupKFold over subjects) static prediction, per
true family (QLearning/CompareToThreshold/LossCounting). Companion to
stage4a_baseline_initial_param_scatter.py, identical panel layout for direct comparison.

Prediction target/method: GRU session-blind's subject embedding (run 1xeoeclu, none/D4)
fed through a within-family GroupKFold + LinearRegression + cross_val_predict against
the session-MEAN of each parameter, evaluated here against the true SESSION-0 value
specifically (matching the baseline scatter's x-axis exactly).

No winsorization ceiling is applied to the GRU predictions -- unlike baseline_rl's
degenerate MLE excursions (which can reach 90-400% of true range), GRU's largest
overshoots are much smaller in relative terms. One notable exception worth checking
visually: CompareToThreshold's learn_rate and softmax_inverse_temperature panels show a
single leverage-point outlier (subject with the single largest embedding-norm outlier
across all 200 subjects, ~4.5x the population median) that pulls those two
cross-validated linear predictions to extreme values; even excluding that one subject,
learn_rate R2 stays clearly negative because the bulk of predictions collapse into a
narrow band relative to the true spread (a genuine shrinkage/compression failure, not
purely an outlier artifact).

Inputs: stage4a_gru_initial_param_scatter.csv (frozen: true_family, subject_id, param,
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
    "QLearning": ["biasL", "choice_kernel_relative_weight", "forget_rate_unchosen",
                  "learn_rate", "softmax_inverse_temperature"],
    "CompareToThreshold": ["biasL", "learn_rate", "softmax_inverse_temperature", "threshold"],
    "LossCounting": ["biasL", "loss_count_threshold_mean", "loss_count_threshold_std"],
}
PARAM_LABEL = {
    "biasL": "biasL", "choice_kernel_relative_weight": "choice-kernel wt",
    "forget_rate_unchosen": "forget rate (unchosen)", "learn_rate": "learn rate",
    "softmax_inverse_temperature": "inverse temp.", "threshold": "threshold",
    "loss_count_threshold_mean": "loss-count thresh (mean)",
    "loss_count_threshold_std": "loss-count thresh (std)",
}
FAM_SHORT = {"QLearning": "QLearning", "CompareToThreshold": "CompareToThreshold",
             "LossCounting": "LossCounting"}


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
            gp = df[(df.true_family == fam) & (df.param == p)]
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

    fig.suptitle("Stage 4a \u2014 GRU session-blind: true initial (session-0) parameter vs "
                "within-family CV prediction", fontsize=12, y=1.05)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scatter-csv", default="stage4a_gru_initial_param_scatter.csv")
    ap.add_argument("--out", default="stage4a_gru_initial_param_scatter.png")
    a = ap.parse_args()

    df = pd.read_csv(a.scatter_csv)
    make_figure(df, a.out)
    print("wrote", a.out)
