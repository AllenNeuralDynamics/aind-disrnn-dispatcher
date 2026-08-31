#!/usr/bin/env python
"""r9 companion figure: P(switch | previous 3 trials), the three classical RL baselines.

r9 tables (d)-(f) already report the RL baselines' generative behavioral match as scalars.
This draws the model-vs-animal scatter behind the history-pattern column of table (e): one
panel per baseline, one dot per 3-back history pattern (abstract encoding, 32 patterns),
x = the mouse's switch probability, y = the rolled-out model's, both subject-mean over the
614 fitted mice with SEM error bars.

Why a separate producer. The rollouts ran once under study 05's `generative-rl-baseline`
variant (2026-07-14/15) and the figure-writing step was skipped there — see that variant's
notes.md "The pattern scatter was recovered later, not re-run". This script is OFFLINE: it
reads the frozen per-pattern rows committed alongside those runs and never touches W&B, per
AGENTS.md §12.

Panel geometry, aggregation level, annotation contents and the 32-colour map all reproduce
the wrapper's own `_plot_history_pattern_comparison_figure`
(`post_training_analysis/generative_analysis.py`), so these panels can be read side by side
with the `combined/history_pattern_comparison_abstract` media panels logged on every GRU
generative run in `generative-v{1,2}@20260623-18074*`.

That is a presentational match, NOT a quantitative one. Those GRU rollouts predate wrapper
PR #60, so ~17% of their sessions were simulated as the wrong task family (a default
uncoupled-baiting task); these RL rollouts are post-#60 and build the task from each
session's own curriculum. The two sides therefore do not share task-family construction --
see r9's Caveats. Pairing them in one side-by-side claim is what the pending GRU rerun is
for.

Note on the two RMSEs. The box in each panel prints the RMSE **across the 32 pattern rows**
of subject-mean values — the quantity the wrapper's own panel annotates, and the one stored
as `quantitative_summary.subject_mean.abstract["3"].rmse`. That is NOT the RMSE column of
r9's table (e), which is sqrt of the *subject-balanced* MSE (per-mouse deltas averaged over
patterns, then across mice). Both are correct; they answer different questions and they rank
the three models differently. r9's "Provenance" section spells this out.

Reproduce: python studies/01-gru-scaling-law/analysis/pswitch_history_patterns.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style  # noqa: E402

# Frozen per-pattern rows, produced by study 05's
# variants/generative-rl-baseline/extract_history_patterns.py (each carries its own _meta
# with the source sha256 and the baseline's W&B run id).
RL_DIR = (STUDY.parent / "05-disrnn-scaling-law" / "variants" / "generative-rl-baseline"
          / "rl_rollout_summaries")
MODELS = [("ctt", "Compare-to-threshold"), ("bari", "Bari 2019"), ("hattori", "Hattori 2019")]
PATTERN_TYPE, N_BACK = "abstract", "3"
OUT_STEM = HERE / "fig_pswitch_history3_rl"

# GRU counterpart, fetched from Beaker by fetch_gru_history_patterns.py. v2 = session
# conditioning ACTIVE (v1 declares scalar encoding but never activates it); H=128 is the only
# hidden size with generative rollouts; D=614 is subject_ratio=1.0; seed 0 of 3.
GRU_FILE = HERE / "gru_history_patterns" / "v2_d614_s0_combined_history_patterns.json"
GRU_LABEL = "GRU v2 (H=128, D=614)"
COMPARE_STEM = HERE / "fig_pswitch_history3_gru_vs_rl"


def wrapper_pattern_colors(patterns: list[str]) -> dict[str, tuple]:
    """Reproduce generative_analysis._build_history_pattern_color_map (tab20->20b->20c)."""
    colors = list(plt.cm.tab20(np.linspace(0, 1, min(20, len(patterns)))))
    if len(patterns) > len(colors):
        colors.extend(plt.cm.tab20b(np.linspace(0, 1, min(20, len(patterns) - len(colors)))))
    if len(patterns) > len(colors):
        colors.extend(plt.cm.tab20c(np.linspace(0, 1, len(patterns) - len(colors))))
    return {p: colors[i % len(colors)] for i, p in enumerate(patterns)}


def draw_panel(ax, rows, summary, patterns, cmap, lim, title) -> None:
    """One model-vs-animal panel: 32 dots + SEM, identity line, r / RMSE / n box."""
    ax.plot(lim, lim, ls="--", color="0.55", lw=1.4, zorder=1)
    for pattern in patterns:
        row = rows[pattern]
        ax.errorbar(row["animal_mean"], row["simulated_mean"],
                    xerr=row["animal_sem"], yerr=row["simulated_sem"],
                    marker="o", markersize=8, color=cmap[pattern], capsize=2.5,
                    elinewidth=1.2, alpha=0.9, lw=0, zorder=3)
    # n (= 32 patterns, identical in every panel) is dropped from the box: it reads as a
    # sample size but is just the dot count, which the reader can see.
    ax.text(0.04, 0.96,
            f"r = {summary['correlation']:.3f}\nRMSE = {summary['rmse']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=14,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "0.7", "alpha": 0.9})
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=10)
    ax.set_xlabel("Mice data")


def main() -> None:
    panels = {}
    for alias, _ in MODELS:
        blob = json.loads((RL_DIR / f"{alias}_history_patterns.json").read_text())
        panel = blob["subject_aggregate"][PATTERN_TYPE][N_BACK]
        panels[alias] = {
            "rows": {r["pattern"]: r for r in panel["rows"] if int(r["n_subjects"]) > 0},
            "summary": panel["summary"],
            "meta": blob["_meta"],
        }

    patterns = sorted(panels["ctt"]["rows"])
    for alias, _ in MODELS:
        assert sorted(panels[alias]["rows"]) == patterns, f"{alias}: pattern set differs"
    cmap = wrapper_pattern_colors(patterns)

    coords = [panels[a]["rows"][p][k] for a, _ in MODELS for p in patterns
              for k in ("animal_mean", "simulated_mean")]
    lim = (0.0, float(np.ceil(max(coords) * 20) / 20) + 0.05)

    apply_presentation_style()
    # Keep SVG text as <text>, not outlined paths, so labels stay editable in Inkscape.
    # (Han, 2026-08.) Not in plot_style because that helper is shared with PNG-only figures.
    # The Helvetica-first font stack itself comes from apply_presentation_style.
    plt.rcParams["svg.fonttype"] = "none"
    # A committed figure must regenerate byte-identically (posthoc-reporting: "idempotent
    # regeneration"). matplotlib otherwise salts SVG element ids per process and stamps a
    # <dc:date>, which churns ~1450 lines on every re-run; pinning the salt and dropping
    # the date leaves only real content in the diff.
    plt.rcParams["svg.hashsalt"] = "pswitch_history3_rl"

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.2))
    for ax, (alias, label) in zip(axes, MODELS):
        draw_panel(ax, panels[alias]["rows"], panels[alias]["summary"],
                   patterns, cmap, lim, label)
    axes[0].set_ylabel("Model")

    handles = [plt.Line2D([], [], marker="o", ls="", markersize=7, color=cmap[p])
               for p in patterns]
    fig.legend(handles=handles, labels=patterns, loc="lower center", ncol=8, frameon=False,
               fontsize=12, handletextpad=0.3, columnspacing=1.1,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("P(switch | previous 3 trials) — per-mouse classical RL models, "
                 "generative rollout, 614 mice", y=0.985)
    fig.tight_layout(rect=(0, 0.27, 1, 0.94))
    fig.savefig(f"{OUT_STEM}.png")
    fig.savefig(f"{OUT_STEM}.svg", metadata={"Date": None})
    print(f"wrote {OUT_STEM}.png / .svg")

    for alias, label in MODELS:
        s = panels[alias]["summary"]
        sems = [panels[alias]["rows"][p][k] for p in patterns
                for k in ("animal_sem", "simulated_sem")]
        print(f"  {label:22s} r={s['correlation']:.4f}  RMSE(across {int(s['n_rows'])} rows)"
              f"={s['rmse']:.4f}  SEM per dot: median={np.median(sems):.4f} "
              f"max={max(sems):.4f}  [source sha256 {panels[alias]['meta']['source_sha256'][:12]}]")

    # --- figure 2: the GRU counterpart alongside the three RL baselines --------------
    if not GRU_FILE.exists():
        print(f"skipped {COMPARE_STEM.name}: {GRU_FILE.name} absent "
              f"(run fetch_gru_history_patterns.py)")
        return
    gru_blob = json.loads(GRU_FILE.read_text())
    gru_panel = gru_blob["subject_aggregate"][PATTERN_TYPE][N_BACK]
    gru = {"rows": {r["pattern"]: r for r in gru_panel["rows"] if int(r["n_subjects"]) > 0},
           "summary": gru_panel["summary"], "meta": gru_blob["_meta"]}
    assert sorted(gru["rows"]) == patterns, "GRU pattern set differs from the RL panels"

    coords2 = coords + [gru["rows"][p][k] for p in patterns
                        for k in ("animal_mean", "simulated_mean")]
    lim2 = (0.0, float(np.ceil(max(coords2) * 20) / 20) + 0.05)

    fig2, axes2 = plt.subplots(1, 4, figsize=(19.5, 6.2))
    draw_panel(axes2[0], gru["rows"], gru["summary"], patterns, cmap, lim2, GRU_LABEL)
    for ax, (alias, label) in zip(axes2[1:], MODELS):
        draw_panel(ax, panels[alias]["rows"], panels[alias]["summary"],
                   patterns, cmap, lim2, label)
    axes2[0].set_ylabel("Model")

    fig2.legend(handles=handles, labels=patterns, loc="lower center", ncol=11, frameon=False,
                fontsize=12, handletextpad=0.3, columnspacing=1.1,
                bbox_to_anchor=(0.5, 0.005))
    fig2.suptitle("P(switch | previous 3 trials) — population GRU vs per-mouse classical RL, "
                  "generative rollout, 614 mice", y=0.985)
    # The asymmetry is on the figure, not only in the report: the GRU rollout predates
    # wrapper #60 and the RL rollouts do not, so this is a provisional comparison.
    fig2.text(0.5, 0.945, "provisional — GRU rollout predates wrapper #60 "
                          "(~17% of sessions simulated as the wrong task family); "
                          "RL rollouts are post-#60",
              ha="center", va="top", fontsize=12, color="#8c3b3b")
    fig2.tight_layout(rect=(0, 0.22, 1, 0.90))
    fig2.savefig(f"{COMPARE_STEM}.png")
    fig2.savefig(f"{COMPARE_STEM}.svg", metadata={"Date": None})
    print(f"wrote {COMPARE_STEM}.png / .svg")
    gs = gru["summary"]
    gsems = [gru["rows"][p][k] for p in patterns for k in ("animal_sem", "simulated_sem")]
    print(f"  {GRU_LABEL:22s} r={gs['correlation']:.4f}  RMSE(across {int(gs['n_rows'])} rows)"
          f"={gs['rmse']:.4f}  SEM per dot: median={np.median(gsems):.4f} "
          f"max={max(gsems):.4f}  [beaker {gru['meta']['beaker_dataset']}]")


if __name__ == "__main__":
    main()
