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
        ("stage2_recovery_vs_baseline.png",
         "**Stage 2 — mild drift.** Fit quality relative to ground truth (a, all near ceiling), and PER-SESSION parameter recovery R² — how well each model's per-session prediction tracks the true drifting parameter — as the mean over parameters vs #subjects (b) and per-parameter at n=200 (c), for baseline_rl, GRU session-blind, and GRU session-conditioned (markers: baseline = square, GRU = circle; color: light blue = 4-d subject-only, dark blue = 4-d + session conditioning, black = baseline). Fixed per-subject estimates (baseline, session-blind) are broadcast to every session; only the session-conditioned GRU predicts a per-session value, and it recovers best (0.84–0.91) while the two fixed models trail together (0.76–0.85) — all three are moderate-to-high because the per-session parameter is dominated by the subject centroid, so session conditioning adds the drift-tracking edge. (Recovery of the drift POSITION itself, where session-blind is 0 by construction, is a separate story — see stage2_session_trajectory.png.) baseline_rl softmax-temperature uses a ROBUST R² (fitted inverse-temperature winsorized at 20, true ceiling ~18.6; Spearman 0.89–0.93). Single seed (42) per cell — no error bars."),
        ("stage2_session_trajectory.png",
         "**Stage 2 — session trajectory.** Per-session parameter recovery R² at Stage 2 (n=200) for "
         "baseline_rl / GRU session-blind / GRU session-conditioned (a) — reads the same source values "
         "as the combined figure's panel c, so the bars agree exactly; session-position recovery, "
         "session-conditioned vs subject-only (b) — subject-only is 0 by construction, session "
         "conditioning recovers it at 0.94; embedding-space drift paths for 8 example subjects, colored "
         "by session position (c) — reconstructed via the training code's own "
         "`compute_session_conditioned_context_dataframe`, frozen to CSV once. Color: black = baseline, "
         "light blue = session-blind/subject-only, dark blue = session-conditioned; in (c), viridis = "
         "session position. Offline from committed CSVs."),
        ("stage2b_likelihood_flip.png",
         "**Stage 2b — the baseline flip.** Static Q-learning collapses (0.939) under "
         "extrapolation while both GRUs stay >0.987 (a); (b) where model separation now lives."),
        ("stage2b_session_trajectory.png",
         "**Stage 2b.** Stronger, non-monotonic drift: session position is harder to recover "
         "(0.94→0.47) but the session delta still carries it; (c) drift paths."),
        ("stage3_recovery_combined.png",
         "**Stage 3 — QL-variant mixture.** Embedding-space PCA colored by true type (a), biasL "
         "(b), learn_rate (c); type decoded at 97.5–99.5% (d), confusion (e), within-family "
         "parameter recovery (f). Model TYPE → cluster; PARAMETERS → position within — but "
         "(b)/(c)'s within-cluster gradient strength is a 2D-PCA-projection artifact, not a "
         "measure of recovery quality: each type's true biasL-encoding direction in the full "
         "4-d embedding aligns with the global top-2 PCA axes by chance (RescorlaWagner's "
         "biasL direction is 82% aligned with local PC1, vs 7–11% for Bari/Hattori), so it "
         "looks clean for RescorlaWagner and scattered for Bari/Hattori despite (f)'s R2 "
         "being uniformly high (0.89–0.96) for all three. (f) is the authoritative recovery "
         "number; (b)/(c) only show what any single 2D view happens to catch."),
        ("stage3_baseline_vs_gru_confusion.png",
         "**Stage 3 — model-identity confusion, baseline vs GRU.** GRU embedding decoding (a, "
         "98.5% correct) vs fixed-baseline model selection (b, 62.5% correct, n=200, MATCHED "
         "comparison): fit exactly the 3 true generative fitters (Bari/Hattori/RescorlaWagner) "
         "per subject; CompareToThreshold, which has no matching true preset in this stage, is "
         "dropped entirely rather than left in as a 4th off-target competitor. Accuracy rises "
         "from 51.0% (4-way, CTT still competing) to 62.5% once CTT stops siphoning off "
         "Bari/Hattori subjects as a spurious best fit. RescorlaWagner's own fitter (wandb run "
         "qy9lof3x) still only recovers 8/66 true-RW subjects even with no off-target competitor "
         "(36/66 misassigned to Bari, 22/66 to Hattori) — a genuine weak-identifiability property "
         "of RW's fit, not a CTT-crowding artifact. GRU still wins by a wide margin (98.5%). "
         "Other selectable comparisons via --baselines: historical (Bari/Hattori/CTT, no RW "
         "fitter, 47.0%) and plus_rw (adds RW alongside CTT, 4-way, 51.0%). Real per-subject "
         "data throughout, replacing an earlier version whose baseline panel was synthesized to "
         "match a remembered accuracy number rather than read from real fits."),
        ("stage3_recovery_vs_baseline.png",
         "**Stage 3 — fit quality and per-session parameter recovery, baseline vs GRU.** (a) "
         "relative held-out likelihood — 6 GRU cells (0.978–0.990) vs baseline_rl best-of-4 "
         "model-selection (0.963, now including RescorlaWagner's own fitter) and its "
         "matched-model ceiling (now n=200 across all 3 families, 0.920 — DOWN from 0.958 at "
         "n=134 Bari/Hattori-only, since RW's own matched fit is weak, 0.67 vs Bari/Hattori's "
         "0.75–0.78, even against RW's own true subjects). (b,c) per-session parameter recovery "
         "(baseline_rl / GRU session-blind broadcast a fixed per-subject estimate; "
         "session-conditioned predicts a genuinely per-session value), mean over each "
         "family's params (b) and per-parameter (c). RescorlaWagner NOW has baseline_rl bars "
         "(RW's own fitter, wandb run qy9lof3x) and they are markedly negative (biasL/"
         "learn_rate/epsilon all R2<-1; Spearman 0.26–0.88 still positive), consistent with (a)'s "
         "weak-identifiability finding. Several baseline parameters show weak identifiability "
         "even after winsorizing near-degenerate MLE fits at the true parameter ceiling — "
         "consistent with this family's previously-reported within-subject session-mean R². GRU "
         "recovers every parameter session-conditioned > session-blind > baseline. Single seed "
         "(42) per cell — no error bars.\n\n"
         "Why is GRU session-blind (also one static value per subject) so much better than "
         "baseline_rl at the SAME task? Both broadcast a fixed per-subject value — neither "
         "tracks drift — so this gap is an estimation story, not a drift-tracking story. "
         "baseline_rl fits each subject independently via differential evolution with no "
         "cross-subject sharing, so weakly-identifiable parameters (forget_rate_unchosen, "
         "learn_rate_rew, softmax_inverse_temperature, choice_kernel_relative_weight) can hit "
         "degenerate boundary values per subject. GRU's embedding is decoded by a readout "
         "trained jointly across all 200 subjects, which regularizes/shrinks toward the "
         "cohort — it cannot run off to those same extremes. biasL is the control case: "
         "strongly identifiable from one subject alone, so baseline_rl and GRU session-blind "
         "are near-tied there (Bari 0.75 vs 0.72; Hattori 0.76 vs 0.78) — the gap opens "
         "specifically where cross-subject pooling helps most and independent per-subject "
         "MLE is most exposed."),
        ("stage3_baseline_initial_param_scatter.png",
         "**Stage 3 — baseline_rl: true initial (session-0) parameter vs fitted static "
         "value.** Isolates whether the negative per-session recovery R2 above comes from "
         "drift-tracking failure (baseline_rl broadcasts one static estimate across all 40 "
         "sessions, structurally cannot track drift) or a genuine point-estimate fit-quality "
         "problem already present at session 0, before drift accumulates. Most weakly-recovered "
         "parameters (choice-kernel weight, forget-rate-unchosen, RescorlaWagner's own three "
         "params) already show ceiling-hugging degenerate MLE fits at session 0 -- red triangles "
         "mark winsorized fits -- so this is at least partly a static fit-quality problem, not "
         "purely drift confusion. biasL is the clean exception (R2=0.74 for Bari/Hattori). "
         "Several parameters keep strong Spearman rank correlation even when R2 is negative "
         "(e.g. RW biasL rho=0.93 vs R2=-0.19, winsorized R2=0.33), pointing to a scale/"
         "calibration miscalibration rather than pure noise for those cases."),
        ("stage3_gru_initial_param_scatter.png",
         "**Stage 3 — GRU session-blind: true initial (session-0) parameter vs "
         "within-family CV prediction.** Same session-0 true value as the baseline scatter "
         "above, but the y-axis is GRU session-blind's held-out prediction (5-fold "
         "GroupKFold over subjects -- every prediction comes from a fold that excluded that "
         "subject) instead of baseline_rl's per-subject independent MLE. A different failure "
         "mode from baseline_rl's ceiling-hugging degenerate fits: GRU predictions "
         "occasionally overshoot the true plausible range too, but only mildly -- at "
         "most ~48% of the parameter's own range (Bari's choice-kernel weight) -- versus "
         "baseline_rl's degenerate MLE excursions, which reach 90-400% of range (e.g. "
         "Bari's softmax_inverse_temperature fitted to 100 against a true ceiling of 15); "
         "no winsorization is applied to the GRU points because none of its overshoots are "
         "of that same catastrophic-outlier character. Separately, two parameters (Bari "
         "and RescorlaWagner's learn_rate) show a genuine systematic "
         "shrinkage bias instead (predictions run ~0.25 too high on average, compressed "
         "variance) rather than outlier-driven negative R2. Bari's forget_rate_unchosen is "
         "essentially unrecovered by either estimator (rho~0, R2<0 for both)."),

        ("stage3_trajectory_recovery.png",
         "**Stage 3 — true parameter drift vs baseline/GRU recovery, one representative "
         "subject per family.** Qualitative complement to the aggregate R2 numbers above: for "
         "one well-fit representative subject per true family, plots the true drifting "
         "parameter across all 40 sessions against baseline_rl (flat static line), GRU "
         "session-blind (flat static line), and GRU session-conditioned (a genuine per-session "
         "curve, reconstructed the same way as the stage2/stage2b trajectory figures). Only "
         "session-conditioned tracks the parameter's actual shape -- most visible for "
         "Bari/Hattori's non-monotonic sinusoidal inverse-temperature drift and RescorlaWagner's "
         "linear learn_rate ramp -- while both flat baselines sit pinned at a single value "
         "throughout, including into the held-out tail."),

        ("stage4a_recovery_combined.png",
         "**Stage 4a — family mixture.** Embedding-space PCA separating the three families (a,b); "
         "GRU embedding decodes family at 100% (c) vs 70% fixed-baseline model selection (d)."),
        ("stage4a_persession_recovery.png",
         "**Stage 4a — per-session parameter recovery, baseline vs GRU.** (a) mean per-session "
         "recovery R2 over each family's drifting params: baseline_rl fails for QLearning "
         "(-0.60, correctly-specified generative model notwithstanding) and is flat/negative for "
         "CompareToThreshold (-0.03), while GRU session-conditioning helps every family (0.58, "
         "0.83, 0.83). (b) per-parameter breakdown, including CompareToThreshold's static "
         "threshold (no drift block — shown as a subject-level recovery check, not drift "
         "tracking; excluded from panel a's mean). CompareToThreshold's softmax_inverse_temperature "
         "and learn_rate are weakly identified already at the session-blind level (R2 -0.04, "
         "-1.16) and session-conditioning makes them WORSE, not merely unhelpful (biasL "
         "0.72→0.25, softmax -0.04→-1.85) — a bias/inverse-temperature confound was checked "
         "directly in the baseline_rl fits and ruled out (corr=0.13); 60.5% of CTT's 200 "
         "fitted threshold values also land outside the true range [0.2, 0.6], despite "
         "threshold's own recovery R2 being good (0.79–0.85) — R2 tracks preserved relative "
         "scale across subjects, which survives even biased point estimates. Likely explanation: "
         "the CompareToThreshold agent's lack of a choice-kernel term leaves its likelihood "
         "surface less constrained for the DE optimizer than QLearning/LossCounting have, not "
         "something specific to session-conditioning. biasL recovery drops under conditioning "
         "for BOTH QLearning (0.67→0.55) and CompareToThreshold (0.72→0.25) — a modest "
         "drop is not unique to CTT, but the magnitude is ~4x larger, and CTT's softmax drop is "
         "far more severe still (delta -1.81, the largest conditioning-induced degradation in "
         "the figure). learn_rate and threshold both improve with conditioning in every family "
         "that has them. Net: conditioning helps on average but a real minority of cells "
         "(both biasL columns, CTT's already-weak softmax) get worse, most severely where the "
         "underlying signal was already marginal."),
        ("stage4a_baseline_initial_param_scatter.png",
         "**Stage 4a — baseline_rl: true initial (session-0) parameter vs fitted static "
         "value.** Direct analog of the Stage-3 baseline scatter, restricted to the diagonal "
         "(each family's fixed fitter scored only against subjects whose true generative "
         "family matches it). Unlike Stage 3's missing-RW-fitter gap, every Stage-4a family "
         "has a correctly-specified baseline -- CompareToThreshold's is genuinely decent "
         "across all 4 parameters after winsorizing degenerate MLE excursions (R2=0.42-0.83, "
         "with 8/67 subjects' learn_rate fits clipped at the true ceiling) -- comparable "
         "in relative frequency but far less severe than QLearning's, which shows the "
         "same severe ceiling-hugging degeneracy Stage 3 found for Bari2019 "
         "(choice_kernel_relative_weight, forget_rate_unchosen, softmax_inverse_temperature "
         "all pinned at the winsor ceiling for a substantial fraction of subjects). "
         "LossCounting's own fit is not uniformly clean either: loss_count_threshold_std "
         "shows a severe raw R2=-2.31 (winsorized -0.31, 31/66 subjects clipped) -- the "
         "worst negative R2 of any Stage-4a baseline parameter, comparable to QLearning's "
         "worst cases despite LossCounting otherwise looking well-behaved on biasL and "
         "loss_count_threshold_mean."),

        ("stage4a_gru_initial_param_scatter.png",
         "**Stage 4a -- GRU session-blind: true initial (session-0) parameter vs "
         "within-family CV prediction.** Companion figure using GRU's held-out (5-fold "
         "GroupKFold over subjects) linear-readout prediction instead of baseline_rl's "
         "per-subject MLE. CompareToThreshold's learn_rate and softmax_inverse_temperature "
         "panels are the one place in this comparison where the correctly-specified "
         "baseline_rl fitter beats the GRU embedding (baseline R2=0.42/0.72 vs GRU "
         "R2=-2.20/-0.13). The two negative R2s are not the same failure: "
         "softmax_inverse_temperature's is almost entirely a single leverage point -- one "
         "subject with the largest embedding-norm outlier across all 200 subjects (~4.5x "
         "the population median) that single-handedly wrecks a 5-fold linear fit; excluding "
         "that one subject alone recovers R2=0.62, essentially matching baseline_rl. "
         "learn_rate's negative R2 is not primarily an outlier artifact -- even excluding the "
         "same subject it stays clearly negative (R2=-0.58) because the remaining 66 "
         "predictions collapse into a narrow band (SD 0.11 vs true SD 0.25), a genuine "
         "variance-shrinkage failure of the cross-validated linear readout on this specific "
         "family/parameter, not a single bad point."),

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
