#!/usr/bin/env python
"""Stage-3 per-session parameter DRIFT trajectory: for one well-fit representative
subject per true family (Bari2019/Hattori2019/RescorlaWagner), plots the true drifting
parameter trajectory against baseline_rl's static fit, GRU session-blind's static
prediction, and GRU session-conditioned's per-session prediction.

Complements stage3_recovery_vs_baseline.py's aggregate R2 numbers with a qualitative,
single-subject view of WHY session-conditioning helps: baseline_rl and GRU session-blind
are necessarily flat lines (one static value per subject); only GRU session-conditioned
tracks the parameter as it actually drifts across the 40 sessions. The held-out tail
(grey band) is the extrapolation split used throughout the ladder.

Representative subjects are chosen as the first (alphabetically) subject per family
whose baseline_rl fit lands within the true parameter's plausible range on every drifting
parameter (same WINSOR ceiling convention as stage3_baseline_initial_param_scatter.py) --
this avoids illustrating the qualitative story with a subject whose static fit is itself
a degenerate MLE outlier.

Inputs (all frozen, committed, offline):
  stage3_full_trajectory.csv        per-(subject,session) true drifting parameters
  stage3_baseline_fits.json         baseline_rl fitted_params_per_subject, keyed by family
  stage3_gru_blind_predictions.csv  GRU session-blind per-subject within-family CV prediction
  stage3_gru_cond_predictions.csv   GRU session-conditioned per-(subject,session) CV prediction

Offline: reads committed CSV/JSON inputs, no network needed.
"""
import argparse
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY_DRIFT_PARAMS = {
    "Bari2019": ["biasL", "learn_rate", "softmax_inverse_temperature"],
    "Hattori2019": ["biasL", "learn_rate_rew", "softmax_inverse_temperature"],
    "RescorlaWagner": ["biasL", "learn_rate", "epsilon"],
}
PARAM_LABEL = {
    "biasL": "biasL", "learn_rate": "learn rate", "learn_rate_rew": "learn rate (rew)",
    "softmax_inverse_temperature": "inverse temp.", "epsilon": "epsilon",
}
WINSOR = {
    "biasL": 1.5, "learn_rate": 0.9, "softmax_inverse_temperature": 15.0,
    "learn_rate_rew": 0.9, "epsilon": 0.35,
}
COL_TRUE = "0.35"
COL_BASE = "black"
COL_BLIND = "#6baed6"
COL_COND = "#08519c"


def pick_well_fit_subject(traj, fits, family):
    subs = sorted(traj[traj.preset_name == family].subject_id.unique())
    for sid in subs:
        fp = fits[family].get(sid, {}).get("fitted_params", {})
        ok = True
        for p in FAMILY_DRIFT_PARAMS[family]:
            if p not in fp:
                ok = False; break
            thr = WINSOR.get(p)
            if thr is not None and abs(fp[p]) > thr * 1.05:
                ok = False; break
        if ok:
            return sid
    return subs[0]


def make_figure(traj, fits, blind, cond, out_png, chosen=None):
    families = list(FAMILY_DRIFT_PARAMS.keys())
    ncols = 3
    fig, axes = plt.subplots(len(families), ncols, figsize=(3.4 * ncols, 3.0 * len(families)))

    chosen = chosen or {}
    for i, fam in enumerate(families):
        params = FAMILY_DRIFT_PARAMS[fam]
        sid = chosen.get(fam) or pick_well_fit_subject(traj, fits, fam)
        sub_traj = traj[traj.subject_id == sid].sort_values("session_index_0based")
        sub_cond = cond[cond.subject_id == sid].copy()
        sub_cond["session_index_0based"] = sub_cond.session_id.str.extract(r"_s(\d+)$")[0].astype(int)
        sub_cond = sub_cond.sort_values("session_index_0based")
        base_fp = fits[fam].get(sid, {}).get("fitted_params", {})
        blind_row = blind[blind.subject_id == sid]
        blind_fp = blind_row.iloc[0] if len(blind_row) else None

        for j, p in enumerate(params):
            ax = axes[i, j]
            col = f"param_{p}"
            x = sub_traj.session_index_0based.values
            y_true = sub_traj[col].values
            is_eval = sub_traj.is_eval.values.astype(bool)

            ax.plot(x, y_true, color=COL_TRUE, lw=1.3, zorder=2)
            ax.scatter(x[~is_eval], y_true[~is_eval], s=14, color=COL_TRUE, zorder=3)
            ax.scatter(x[is_eval], y_true[is_eval], s=20, color=COL_TRUE, marker="D",
                       edgecolor="white", lw=0.5, zorder=3)

            if p in base_fp:
                ax.axhline(base_fp[p], color=COL_BASE, lw=1.6, ls="-", zorder=1)
            if blind_fp is not None and f"pred_{p}" in blind_fp.index and not pd.isna(blind_fp[f"pred_{p}"]):
                ax.axhline(blind_fp[f"pred_{p}"], color=COL_BLIND, lw=1.4, ls="--", zorder=1)

            pcol = f"pred_{p}"
            if pcol in sub_cond.columns:
                valid = ~sub_cond[pcol].isna()
                ax.plot(sub_cond.session_index_0based[valid], sub_cond[pcol][valid],
                        color=COL_COND, lw=1.4, zorder=4)

            if is_eval.sum() > 0:
                eval_start = x[is_eval].min()
                ax.axvspan(eval_start - 0.5, x.max() + 0.5, color="0.92", zorder=0)

            ax.set_title(f"{fam.replace('2019','')}: {PARAM_LABEL[p]} (subject {sid})",
                        fontsize=8.5, loc="left")
            ax.set_xlabel("session index", fontsize=8)
            ax.set_ylabel(PARAM_LABEL[p], fontsize=8)
            ax.margins(y=0.08)
            for s in ax.spines.values():
                s.set_visible(False)

    handles = [
        plt.Line2D([], [], color=COL_TRUE, lw=1.3, marker="o", ms=4, label="true (drifting, train)"),
        plt.Line2D([], [], color=COL_TRUE, lw=0, marker="D", ms=5, label="true (held-out tail)"),
        plt.Line2D([], [], color=COL_BASE, lw=1.6, label="baseline_rl (static fit)"),
        plt.Line2D([], [], color=COL_BLIND, lw=1.4, ls="--", label="GRU session-blind (static)"),
        plt.Line2D([], [], color=COL_COND, lw=1.4, label="GRU session-conditioned (per-session)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3,
               frameon=False, fontsize=8)
    fig.suptitle("Stage 3 \u2014 one representative (well-fit) subject per family: "
                "true parameter drift vs baseline/GRU recovery", fontsize=12, y=1.09)
    fig.tight_layout(rect=[0.01, 0, 1, 0.92])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-csv", default="stage3_full_trajectory.csv")
    ap.add_argument("--baseline-fits-json", default="stage3_baseline_fits.json")
    ap.add_argument("--gru-blind-csv", default="stage3_gru_blind_predictions.csv")
    ap.add_argument("--gru-cond-csv", default="stage3_gru_cond_predictions.csv")
    ap.add_argument("--out", default="stage3_trajectory_recovery.png")
    a = ap.parse_args()

    traj = pd.read_csv(a.trajectory_csv)
    fits = json.load(open(a.baseline_fits_json))
    blind = pd.read_csv(a.gru_blind_csv)
    cond = pd.read_csv(a.gru_cond_csv)

    make_figure(traj, fits, blind, cond, a.out)
    print("wrote", a.out)
