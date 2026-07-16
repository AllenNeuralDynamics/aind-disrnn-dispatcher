#!/usr/bin/env python
"""Stage-2b SESSION-TRAJECTORY recovery (panels a,b,c), house-styled to match stage 1/2.

  a  Per-session PARAMETER recovery R2 at Stage 2b, n=200, THREE models (baseline_rl /
     GRU session-blind / GRU session-conditioned) — the exact stage2_recovery_figure.py
     panel-c format (same colors: black/light-blue/dark-blue, same bar layout), for direct
     side-by-side comparison with the Stage-2 combined figure.
  b  Session-POSITION recovery (sessfrac R2): how well the embedding locates a session
     within its subject, Stage 2 vs Stage 2b, subject-only vs session-conditioned.
     Subject-only is 0.00 by construction (a fixed per-subject point carries no session
     index); session conditioning recovers it (S2 0.94 -> S2b 0.47 as the drift turns
     non-monotonic). baseline_rl has no session-position estimate (fixed per-subject fit
     only) so it is not shown in panel b.

  c  Embedding-space drift paths for 8 example subjects (Stage 2b, session-conditioned
     run), colored by session position (viridis, 0->1). PCA via numpy SVD (no sklearn
     dependency), same pattern as recovery_report.py's disRNN panel-C. Black star = first
     session. Reads stage2b_embedding_trajectories.csv, frozen via the training code's own
     reconstruction (utils.multisubject.compute_session_conditioned_context_dataframe --
     the same function stage2_session_traj.py already uses for Stage 2), so no
     reimplementation of the session-delta MLP.

House convention: black = baseline, light blue = session-blind/subject-only, dark blue =
session-conditioned; in panel b, stage is distinguished by saturation (Stage-2 lighter,
Stage-2b full). Square despined panels, presentation-scale fonts. Offline: reads the
committed curated stage2b_trajectory_recovery.csv (frozen; baseline_rl S2b row from
compute_stage2b_baseline_persession.py, same broadcast method as the Stage-2 baseline
recompute) and stage2b_embedding_trajectories.csv (frozen coordinates for panel c).
Single seed (42) per cell -> no error bars.

The panel-c coordinates were fetched once via a one-off HPC job (the only network step in
this whole figure's provenance) and frozen to CSV; this producer itself never calls W&B.
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = {"subj_only": "#6baed6", "conditioned": "#08519c"}
ALPHA = {"S2": 0.55, "S2b": 1.0}
PARAMS = ["biasL", "learn_rate", "softmax_temp"]

# Panel-a house convention identical to stage2_recovery_figure.py panel c: baseline = black,
# session-blind = light blue, session-conditioned = dark blue.
COL = {"baseline_rl": "black", "subj_only": "#6baed6", "conditioned": "#08519c"}
LAB = {"baseline_rl": "baseline_rl", "subj_only": "GRU 4-d, session-blind",
       "conditioned": "GRU 4-d + session cond."}


def _bars_s2b(ax, df, ylabel, title):
    """Stage-2b only, 3 bars/parameter (baseline / session-blind / session-cond) —
    exactly the stage2_recovery_figure.py panel-c format, for direct side-by-side."""
    xpos = np.arange(len(PARAMS)); w = 0.27
    order = ["baseline_rl", "subj_only", "conditioned"]
    for i, cond in enumerate(order):
        row = df[(df.stage == "S2b") & (df.cond == cond)].iloc[0]
        vals = [row[f"r2_{p}"] for p in PARAMS]
        ax.bar(xpos + (i - 1) * w, vals, w, color=COL[cond], label=LAB[cond])
    ax.axhline(1.0, color="0.7", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(xpos); ax.set_xticklabels(["biasL", "learn\nrate", "softmax\ntemp"], fontsize=7)
    ax.set_ylabel(ylabel); ax.set_ylim(0, 1.05); ax.set_title(title)


def _panel_c(ax, emb):
    """Embedding-space drift paths for 8 example subjects, colored by session position.
    PCA via numpy SVD (no sklearn dependency), same pattern as recovery_report.py's
    disRNN panel-C. One path per subject; color = session_phase (0 -> 1)."""
    ecols = [c for c in emb.columns if c.startswith("embedding_")]
    X = emb[ecols].to_numpy(float)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ Vt[:2].T
    emb = emb.copy(); emb["pc1"] = pcs[:, 0]; emb["pc2"] = pcs[:, 1]
    cmap = plt.get_cmap("viridis")
    sc = None
    for sid, g in emb.groupby("subject_id", sort=False):
        g = g.sort_values("session_index")
        ax.plot(g.pc1, g.pc2, color="0.75", lw=1.0, alpha=0.8, zorder=1)
        sc = ax.scatter(g.pc1, g.pc2, c=g.session_phase, cmap=cmap, vmin=0, vmax=1,
                         s=22, edgecolor="white", lw=0.3, zorder=2)
        ax.scatter(g.pc1.iloc[0], g.pc2.iloc[0], marker="*", s=90, c="black",
                   edgecolor="white", lw=0.6, zorder=3)
    ax.set_xlabel("PC1 \u2192"); ax.set_ylabel("PC2 \u2192")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Drift paths, {emb.subject_id.nunique()} example subjects\n(Stage 2b, session-conditioned)")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("session position", fontsize=7); cb.ax.tick_params(labelsize=6)


def make_figure(traj, out_png, emb=None):
    ncols = 3 if emb is not None else 2
    fig = plt.figure(figsize=(9 if emb is None else 13.5, 3.6))
    gs = fig.add_gridspec(1, ncols, wspace=0.5 if emb is not None else 0.42)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- a: per-session parameter recovery, Stage 2b only (matches stage2 panel c) ----
    _bars_s2b(axA, traj, "per-session recovery R\u00b2", "Per-session (Stage 2b, n=200)")
    axA.legend(frameon=False, loc="upper center", fontsize=5.5, ncol=1,
               bbox_to_anchor=(0.5, -0.24), handletextpad=0.4, handlelength=1.2)

    # ---- b: session-position recovery ----
    combos = [("S2", "conditioned"), ("S2", "subj_only"), ("S2b", "conditioned"), ("S2b", "subj_only")]
    labs = ["S2\ncond.", "S2\nsubj-only", "S2b\ncond.", "S2b\nsubj-only"]
    for x, (stage, cond) in enumerate(combos):
        v = traj[(traj.stage == stage) & (traj.cond == cond)].sessfrac_R2.iloc[0]
        axB.bar(x, v, color=BLUE[cond], alpha=ALPHA[stage])
        axB.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(range(4)); axB.set_xticklabels(labs, fontsize=6.5)
    axB.set_ylabel("R\u00b2 recovering session position"); axB.set_ylim(0, 1.05)
    axB.set_title("Session-position recovery")

    axes = [axA, axB]
    if emb is not None:
        axC = fig.add_subplot(gs[0, 2])
        _panel_c(axC, emb)
        axC.set_box_aspect(1)
        axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)
        axes.append(axC)

    for ax in axes:
        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ax, l in zip(axes, "abc"):
        ax.text(-0.18, 1.04, l, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-csv", default="stage2b_trajectory_recovery.csv")
    ap.add_argument("--emb-csv", default="stage2b_embedding_trajectories.csv",
                     help="Per-(subject,session) embedding coords for example subjects; "
                          "adds panel c (drift paths). Pass empty string to omit.")
    ap.add_argument("--out", default="stage2b_session_trajectory.png")
    a = ap.parse_args()
    emb_df = pd.read_csv(a.emb_csv) if a.emb_csv else None
    make_figure(pd.read_csv(a.traj_csv), a.out, emb=emb_df)
    print("wrote", a.out)
