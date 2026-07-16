#!/usr/bin/env python
"""Stage-4a baseline_rl: true INITIAL (session-0, pre-drift) parameter vs baseline_rl's
per-subject independent-MLE fitted static value, per true family (QLearning/
CompareToThreshold/LossCounting). Direct analog of
stage3_initial_param_scatter.py, extended to Stage 4a's per-family parameter set.

Unlike Stage 3, Stage 4a has a correctly-specified fixed baseline for EVERY true family
(no missing-fitter problem). Restricted to the "diagonal": each fitter's fits are only
scored against subjects whose TRUE generative family matches that fitter (baseline_rl's
model-selection run fits every subject with every family's fixed model for the confusion
matrix, but the true-parameter comparison here is only meaningful when fitter family ==
subject's actual generative family).

Winsorization ceilings (WINSOR_4A) mirror the true generative parameter ranges from
hierarchical_rl_stage4a.yaml, same convention as Stage 3's scatter and
compute_stage4a_persession_recovery.py.

Inputs: stage4a_baseline_initial_param_scatter.csv (frozen: true_family, subject_id,
param, true_initial, baseline_fitted_static).

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
WINSOR = {
    "biasL": 1.5, "learn_rate": 0.9, "softmax_inverse_temperature": 15.0,
    "loss_count_threshold_mean": 6.0, "threshold": 0.6,
    "forget_rate_unchosen": 0.4, "choice_kernel_relative_weight": 0.5,
    "loss_count_threshold_std": 2.0,
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
            gp = df[(df.true_family == fam) & (df.param == p)]
            x = gp.true_initial.values
            y = gp.baseline_fitted_static.values
            r2_raw = r2_score(x, y)
            thr = WINSOR.get(p)
            if thr is not None:
                clipped = y > thr
                y_plot = np.clip(y, None, thr)
                r2_w = r2_score(x, y_plot)
            else:
                clipped = np.zeros(len(y), dtype=bool)
                y_plot = y
                r2_w = r2_raw
            rho = spearmanr(x, y).correlation

            lims = [min(x.min(), y_plot.min()), max(x.max(), y_plot.max())]
            pad = 0.06 * (lims[1] - lims[0] if lims[1] > lims[0] else 1)
            lims = [lims[0] - pad, lims[1] + pad]
            ax.plot(lims, lims, color="0.75", lw=1, ls="--", zorder=1)
            ax.scatter(x[~clipped], y_plot[~clipped], s=18, color="#1f77b4", alpha=0.65,
                       edgecolor="white", lw=0.3, zorder=2)
            if clipped.any():
                ax.scatter(x[clipped], y_plot[clipped], s=22, marker="^", color="#d62728", alpha=0.8,
                           edgecolor="white", lw=0.3, zorder=3)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_box_aspect(1)
            ax.locator_params(axis="both", nbins=4)
            ttl = f"{PARAM_LABEL[p]}\nR\u00b2={r2_raw:.2f} (wins. {r2_w:.2f}), \u03c1={rho:.2f}"
            ax.set_title(ttl, fontsize=8.5, loc="left")
            ax.set_xlabel("true (session 0)", fontsize=8)
            ax.set_ylabel("fitted (static)", fontsize=8, labelpad=2)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i, 0].text(-0.55, 0.5, FAM_SHORT[fam], transform=axes[i, 0].transAxes,
                        fontsize=11, fontweight="bold", va="center", ha="right", rotation=90)

    fig.suptitle("Stage 4a \u2014 baseline_rl: true initial (session-0) parameter vs "
                "fitted static value", fontsize=12, y=1.06)
    handles = [
        plt.Line2D([], [], marker="o", color="#1f77b4", lw=0, label="fit within plausible range"),
        plt.Line2D([], [], marker="^", color="#d62728", lw=0,
                   label="winsorized (degenerate MLE, clipped at true ceiling)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2,
               frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0.03, 0, 1, 0.94])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scatter-csv", default="stage4a_baseline_initial_param_scatter.csv")
    ap.add_argument("--out", default="stage4a_baseline_initial_param_scatter.png")
    a = ap.parse_args()

    df = pd.read_csv(a.scatter_csv)
    make_figure(df, a.out)
    print("wrote", a.out)
