#!/usr/bin/env python
"""Single producer for the 04-gru-vs-disrnn-embedding-recovery study.

Reads the committed curated grids (offline, no W&B needed):
  - analysis/ladder_results.csv           (headline per stage)
  - analysis/stage4b_recovery_grid.csv    (per-run Stage-4b GRU recovery)
  - analysis/disrnn_stage4a_recovery_grid.csv (per-run disRNN Stage-4a recovery)

Writes:
  - analysis/recovery_summary.json  (with a _meta provenance block)
  - analysis/recovery_summary.csv
  - analysis/fig_ladder.png             (the baseline-flip ladder)
  - analysis/fig_disrnn_stage4a.png     (disRNN replication)
and regenerates the <!-- BEGIN result-N --> / <!-- END result-N --> blocks in
reports/r1-gru-ladder.md and reports/r2-disrnn-replication.md.

Idempotent: `make -C studies/04-gru-vs-disrnn-embedding-recovery` twice == empty diff.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # analysis/
STUDY = HERE.parent                              # studies/<study>/
sys.path.insert(0, str(STUDY.parent / "util"))   # studies/util
from _meta import build_meta                      # noqa: E402
from plot_style import apply_presentation_style   # noqa: E402

WANDB_PROJECT = "embedding_recovery"
WANDB_GROUPS = [
    "gru-stage1", "gru-stage2", "gru-stage2b", "gru-stage3", "gru-stage4a", "gru-stage4b",
    "disrnn-stage4a",
    "baseline-rl-stage1", "baseline-rl-stage2", "baseline-rl-stage2b",
    "baseline-bari-stage3", "baseline-hattori-stage3", "baseline-ctt-stage3",
    "baseline-bari-stage4a", "baseline-ctt-stage4a", "baseline-losscounting-stage4a",
    "baseline-bari-stage4b", "baseline-ctt-stage4b", "baseline-losscounting-stage4b",
]
FAMCOL = {"QLearning": "#2166ac", "CompareToThreshold": "#b2182b", "LossCounting": "#1b7837"}


def _pca2(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def main() -> None:
    ladder = pd.read_csv(HERE / "ladder_results.csv", dtype=str)
    s4b = pd.read_csv(HERE / "stage4b_recovery_grid.csv")
    dis = pd.read_csv(HERE / "disrnn_stage4a_recovery_grid.csv")

    # ---- curated summary JSON ----
    summary = {
        "_meta": build_meta("analysis/recovery_report.py", WANDB_GROUPS, study_root=STUDY),
        "wandb_project": WANDB_PROJECT,
        "ladder": ladder.to_dict(orient="records"),
        "stage4b_gru": s4b.to_dict(orient="records"),
        "disrnn_stage4a": dis.to_dict(orient="records"),
    }
    (HERE / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    ladder.to_csv(HERE / "recovery_summary.csv", index=False)

    apply_presentation_style()

    # ---- Figure 1: the baseline-flip ladder ----
    _fig_ladder(ladder)
    # ---- Figure 2: disRNN Stage-4a replication ----
    _fig_disrnn(dis)

    # ---- regenerate report blocks ----
    _update_reports(ladder, s4b, dis)
    print("recovery_report: wrote summary + 2 figures + report blocks")


def _fig_ladder(ladder):
    # relative-LL: GRU vs baseline across the ladder (numeric midpoints of the ranges)
    def mid(s):
        s = str(s).replace("~", "").split("(")[0].strip()
        if "\u2013" in s:
            a, b = s.split("\u2013"); return (float(a) + float(b)) / 2
        try: return float(s)
        except ValueError: return np.nan
    L = ladder[ladder.stage != "4a-disRNN"].copy()
    L["gru"] = L.model_rel_LL.map(mid); L["base"] = L.baseline_rel_LL.map(mid)
    x = np.arange(len(L))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, L.gru, marker="o", color="#1a1a1a", label="GRU (data-driven)")
    ax.plot(x, L.base, marker="s", color="#b2182b", label="correct-model baseline")
    ax.axhline(1.0, color="#cccccc", ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels("S" + L.stage, rotation=0)
    ax.set_ylim(0.35, 1.03); ax.set_xlabel("ladder stage"); ax.set_ylabel("relative likelihood")
    ax.set_title("The baseline flip: correct-model baseline breaks under\nextrapolation + mixture; the GRU embedding does not", fontsize=13, loc="left")
    ax.annotate("baseline flip\n(2b: extrapolation)", xy=(2, L.base.iloc[2]), xytext=(2.1, 0.62),
                fontsize=11, color="#b2182b", arrowprops=dict(arrowstyle="->", color="#b2182b"))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(HERE / "fig_ladder.png"); plt.close(fig)


def _fig_disrnn(dis):
    gru_lik = {4: 0.990552868070854, 8: 0.990559486606228, 16: 0.9881624486414816}
    mk = {"none": "o", "scalar": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axA = axes[0]
    for enc in ["none", "scalar"]:
        sub = dis[dis.enc == enc].sort_values("embed")
        axA.plot(sub.embed, sub.family_acc_subject, marker=mk[enc], color="#762a83", ms=9,
                 mfc="#762a83" if enc == "scalar" else "white", mec="#762a83", mew=1.6, label=f"disRNN ({enc})")
    axA.axhline(1.0, color="#1a1a1a", lw=1.4); axA.text(16.2, 1.0, "GRU=1.00", va="center", fontsize=11)
    axA.axhline(0.70, color="#b2182b", ls="--", lw=1.3); axA.text(16.2, 0.70, "baseline=0.70", va="center", fontsize=11, color="#b2182b")
    axA.axhline(1/3, color="#cccccc", ls=":", lw=1.1); axA.text(16.2, 1/3, "chance", va="center", fontsize=10, color="#999")
    axA.set_xticks([4, 8, 16]); axA.set_xlim(3.4, 22); axA.set_ylim(0.28, 1.06)
    axA.set_xlabel("subject embedding size"); axA.set_ylabel("family-decoding accuracy")
    axA.set_title("0.95\u20130.98 with session cond.;\n0.75\u20130.90 without", fontsize=12, loc="left")
    axA.legend(frameon=False, fontsize=11, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    axB = axes[1]
    x = np.array([4, 8, 16])
    dn = dis[dis.enc == "none"].set_index("embed").reindex(x)
    ds = dis[dis.enc == "scalar"].set_index("embed").reindex(x)
    axB.plot(x, [gru_lik[e] for e in x], marker="D", color="#1a1a1a", label="GRU (none)")
    axB.plot(x, dn.rel_LL.values, marker="o", color="#762a83", mfc="white", mec="#762a83", mew=1.6, label="disRNN (none)")
    axB.plot(x, ds.rel_LL.values, marker="s", color="#762a83", label="disRNN (scalar)")
    axB.set_xticks([4, 8, 16]); axB.set_xlim(3.4, 17)
    axB.set_xlabel("subject embedding size"); axB.set_ylabel("relative likelihood")
    axB.set_title("disRNN costs ~4\u20136 pts likelihood\nfor its sparse latent", fontsize=12, loc="left")
    axB.legend(frameon=False, fontsize=11, loc="center right")

    # Panel C: embedding-space PCA of the scalar-D16 disRNN subject embedding,
    # colored by true model family (the interpretable clusters). Reads the
    # committed emb CSV; the true family per subject is joined from the grid.
    axC = axes[2]
    FAMCOL = {"QLearning": "#2166ac", "CompareToThreshold": "#b2182b", "LossCounting": "#1b7837"}
    embp = HERE / "emb_disrnn_scalar_D16.csv"
    if embp.exists():
        emb = pd.read_csv(embp)
        ecols = [c for c in emb.columns if c.startswith("emb_")]
        X = emb[ecols].to_numpy(float); X = X - X.mean(0)
        # PCA via numpy SVD (sklearn absent in default env)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = X @ Vt[:2].T
        for f, col in FAMCOL.items():
            m = emb.dominant_family.values == f
            axC.scatter(pcs[m, 0], pcs[m, 1], s=26, c=col, edgecolor="white", lw=0.4,
                        label=f.replace("CompareToThreshold", "CompareThresh"), alpha=0.9)
        axC.set_xlabel("PC1 \u2192"); axC.set_ylabel("PC2 \u2192")
        axC.set_xticks([]); axC.set_yticks([])
        axC.legend(frameon=False, fontsize=10, loc="lower left")
        axC.set_title("disRNN embedding separates\nthe three families (scalar D16)", fontsize=12, loc="left")
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "emb CSV not staged\n(panel skipped)", ha="center", va="center",
                 fontsize=10, color="#999", transform=axC.transAxes)
    fig.tight_layout(); fig.savefig(HERE / "fig_disrnn_stage4a.png"); plt.close(fig)




def _replace_block(text: str, n: int, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN result-{n} -->).*?(<!-- END result-{n} -->)", re.DOTALL)
    return pat.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(2), text)


def _update_reports(ladder, s4b, dis):
    # r1 — GRU ladder
    r1 = HERE / "reports" / "r1-gru-ladder.md"
    tbl = "| stage | generator | GRU rel-LL | baseline rel-LL | recovery | value |\n|---|---|---|---|---|---|\n"
    for _, r in ladder[ladder.stage != "4a-disRNN"].iterrows():
        tbl += f"| {r.stage} | {r.generator} | {r.model_rel_LL} | {r.baseline_rel_LL} | {r.recovery} | {r.recovery_value} |\n"
    body1 = ("[regenerated by `analysis/recovery_report.py` — do not edit by hand]\n\n"
             "![baseline-flip ladder](../fig_ladder.png)\n\n"
             "*Relative likelihood (model NL / ground-truth NL, ceiling 1.0) of the GRU vs the "
             "correct-model-class baseline across the ladder. N=200.*\n\n" + tbl +
             "\n- **The baseline flip is the study's spine.** A correctly-specified baseline matches the "
             "GRU on stationary (S1) and interpolable (S2) data, then breaks under extrapolation "
             "(S2b: 0.94 vs GRU >0.987) and under mixed structure (S3 model-selection 47%, S4a 70%), "
             "while the GRU embedding recovers the true structure at 97.5\u2013100%.\n"
             "- **Embedding dimension is the identifiability knob** \u2014 recovery scales with D, not "
             "hidden-unit count; higher-diversity mixtures (S3/S4) need D=16.\n"
             "- **Stage-4b** (per-session family switching): recovery lives at the SUBJECT level "
             "(mixture-weight R\u00b2 0.55 @D16), not the session level \u2014 session-conditioning adds "
             "nothing over subject identity for decoding a session's family (0.62 vs 0.63), because "
             "Dirichlet(0.5) subjects are concentrated (mean dominant weight 0.70).\n")
    # Per-stage recovery / parameter-recovery / embedding-space figures, embedded in
    # order. The S3/S4a "combined" figures already carry the embedding-space PCA
    # panels (colored by true type/family + key params); the S3 standalone
    # embedding-space and baseline-confusion figures remain in figures/ as subsets.
    stage_figs = [
        ("recovery_ground_truth_schematic.png",
         "**Setup.** Each subject occupies a parameter subregion; sessions drift within it. "
         "The model must recover the subject-embedding table (between-subject) and the "
         "session-conditioning MLP (within-subject drift)."),
        ("stage1_recovery_vs_baseline.png",
         "**Stage 1 — static.** Parameter recovery R² vs #subjects (a), fit quality at "
         "ceiling (b), per-parameter R² (c). Embedding size, not network width, is the knob."),
        ("stage2_recovery.png",
         "**Stage 2 — mild drift.** Subject-level parameter recovery R² vs #subjects."),
        ("stage2_likelihood_comparison.png",
         "**Stage 2.** All models sit near the ceiling — likelihood cannot separate them, "
         "which is what motivates the recovery axis."),
        ("stage2_session_trajectory.png",
         "**Stage 2.** Only the session-conditioning MLP encodes drift position "
         "(subject-only delta-zeroed = R² 0.00 by construction); (c) each subject traces a "
         "drift path in embedding space."),
        ("stage2b_likelihood_flip.png",
         "**Stage 2b — the baseline flip.** Static Q-learning collapses (0.939) under "
         "extrapolation while both GRUs stay >0.987 (a); (b) where model separation now lives."),
        ("stage2b_session_trajectory.png",
         "**Stage 2b.** Stronger, non-monotonic drift: session position is harder to recover "
         "(0.94→0.47) but the session delta still carries it; (c) drift paths."),
        ("stage3_recovery_combined.png",
         "**Stage 3 — QL-variant mixture.** Embedding-space PCA colored by true type (a), biasL "
         "(b), learn_rate (c); type decoded at 97.5–99.5% (d), confusion (e), within-family "
         "parameter recovery (f). Model TYPE → cluster; PARAMETERS → position within."),
        ("stage4a_recovery_combined.png",
         "**Stage 4a — family mixture.** Embedding-space PCA separating the three families (a,b); "
         "GRU embedding decodes family at 100% (c) vs 70% fixed-baseline model selection (d)."),
        ("stage4b_recovery.png",
         "**Stage 4b — per-session family switching.** Mixture-weight recovery vs embedding size "
         "(a); subject-vs-session dissociation null (b); per-session family confusion (c)."),
    ]
    gallery = "\n### Per-stage figures\n\n"
    for fn, cap in stage_figs:
        gallery += f"![{fn}](../figures/{fn})\n\n*{cap}*\n\n"
    body1 = body1 + gallery.rstrip("\n")
    r1.write_text(_replace_block(r1.read_text(), 1, body1))
    # r2 — disRNN replication
    r2 = HERE / "reports" / "r2-disrnn-replication.md"
    dtbl = "| enc | D | rel-LL | family-decoding acc |\n|---|---|---|---|\n"
    for _, r in dis.sort_values(["enc", "embed"]).iterrows():
        dtbl += f"| {r.enc} | {r.embed} | {r.rel_LL:.4f} | {r.family_acc_subject:.3f} |\n"
    body2 = ("[regenerated by `analysis/recovery_report.py` — do not edit by hand]\n\n"
             "![disRNN Stage-4a replication](../fig_disrnn_stage4a.png)\n\n"
             "*disRNN family decoding (left) and likelihood cost (right) vs the GRU (=1.00 family, "
             "~0.99 rel-LL) and the correct-baseline ceiling (0.70). N=200, Stage-4a generator.*\n\n" + dtbl +
             "\n- **Interpretability is nearly free for recovery.** The information-bottlenecked disRNN "
             "decodes model family at 0.95\u20130.98 (with session conditioning), nearly matching the GRU's "
             "1.00 and far above the 0.70 baseline ceiling.\n"
             "- **The bottleneck costs ~4\u20136 pts of likelihood** (rel-LL ~0.93\u20130.95 vs GRU ~0.99) \u2014 "
             "smallest for `none` D4 (4.2 pts), largest for `scalar` D4 (6.4 pts).\n"
             "- **Session conditioning matters more for the disRNN.** Without it (`none`), decoding is "
             "lower and more variable (0.75\u20130.90); the GRU's raw embedding already hit 1.00.")
    r2.write_text(_replace_block(r2.read_text(), 1, body2))


if __name__ == "__main__":
    main()
