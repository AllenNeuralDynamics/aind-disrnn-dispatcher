#!/usr/bin/env python
"""Single producer for the beta-scan study's normalized reports.

Per docs/posthoc-analysis.md: this is the ONE script that produces the curated
outputs (JSON + CSV + figures) and regenerates the ``<!-- BEGIN result-N -->``
blocks in analysis/reports/r*.md. Idempotent: ``make -C studies/beta-scan``
twice yields identical files.

WHAT IT SCORES
--------------
The study scans ``update_net_latent_penalty_multiplier`` x base beta x lr x seed
(48 designed / 43 clean) at 100 mice, asking how the multiplier sparsifies the
disRNN's six information bottlenecks and whether that costs held-out transfer.

HONEST SPARSITY METRIC (important — see reports/INDEX.md caveat)
---------------------------------------------------------------
Openness is reported as ``total_openness`` = Sum(1 - sigma) over a family's
channels (absolute, in nats; ~0 = fully closed). We deliberately do NOT headline
``n_eff_open_frac`` (the participation ratio): it is scale-invariant and reports
a spuriously high value when a bottleneck is fully closed (all sigma~1), which
inverts the true ranking (19/43 runs mis-ranked). ``n_eff_open_frac`` is retained
in the raw grid CSV for reference but never drives a conclusion here.

DATA SOURCE
-----------
The settled grid metrics live in the study's committed
``analysis/beta_scan_final_grid.csv`` (one row per clean run: knobs + per-family
threshold-free sparsity metrics + held-out LL), which was produced by the
offline threshold-free backfill (analysis/backfill_sparsity_metrics.py) on the
HPC node against the wrapper commit pinned in ../environment.lock. That CSV is
the source of truth; this script aggregates it into curated per-cell summaries.
(The live W&B project ``disrnn_updnet_bottleneck_ratio_100mice`` remains the
upstream of that CSV; see analysis/backfill_sparsity_metrics.py for the pull.)

OUTPUTS (all committed so the reports render in-repo)
  analysis/beta_scan_summary.json  - curated per-(mult,beta) means/sem/n + _meta
  analysis/beta_scan_summary.csv   - flat per-(mult,beta,family) table
  analysis/fig_bottlenecks_by_mult.png - 6-family openness vs multiplier (by beta)
  analysis/fig_mult_axis_heldout.png   - update-net-latent openness + held-out vs multiplier
  then regenerates reports/r1 (sparsity) and reports/r2 (held-out transfer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))  # studies/util
sys.path.insert(0, str(HERE))  # for sibling update_reports import
from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style, t975  # noqa: E402

GRID_CSV = HERE / "beta_scan_final_grid.csv"
# Settled W&B groups feeding the clean grid (main short-horizon grid + mult=10
# supplement). Kept in sync with the project after the wrap-up run deletion.
WANDB_GROUPS = [
    "updnet-ratio-100mice@20260703-200122",       # main 48-run short-horizon grid (43 clean)
    "updnet-ratio-100mice-mult10-supp@20260706-093606",  # mult=10 supplement (9 clean)
]
WANDB_PROJECT_URL = "https://wandb.ai/AIND-disRNN/disrnn_updnet_bottleneck_ratio_100mice"

FAMILIES = [
    ("latent", "LATENT (recurrent)", 5),
    ("update_net_obs", "UPDATE \u2190 obs (choice+rew)", 10),
    ("update_net_latent", "UPDATE \u2190 latent", 25),
    ("update_net_subj", "UPDATE \u2190 subject emb", 20),
    ("choice_net_latent", "CHOICE \u2190 latent", 5),
    ("choice_net_subj", "CHOICE \u2190 subject emb", 4),
]
MULTS = [1, 2, 5, 10]
# base beta color palette (viridis, weak->strong)
BCOLORS = None  # set in _figures once betas are known


def _read_grid():
    import csv
    rows = []
    with open(GRID_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _stat(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "sem": None, "ci95": None, "n": 0}
    a = np.asarray(vals, float)
    sem = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "ci95": float(t975(n) * sem), "n": n}


def summarize(rows):
    """Curated per-(mult,beta) means for every family + held-out LL."""
    betas = sorted({_f(r["beta"]) for r in rows if _f(r["beta"]) is not None})
    cells = {}
    for m in MULTS:
        for b in betas:
            sub = [r for r in rows if _f(r["mult"]) == m and _f(r["beta"]) == b]
            if not sub:
                continue
            key = f"m{m}_b{b:g}"
            cell = {"heldout_ll": _stat([_f(r["heldout_eval_ll"]) for r in sub])}
            for fam, _lbl, _n in FAMILIES:
                cell[f"{fam}_openness"] = _stat([_f(r[f"{fam}.total_openness"]) for r in sub])
                cell[f"{fam}_frac_half_open"] = _stat([_f(r[f"{fam}.frac_open_s0p5"]) for r in sub])
            cells[key] = cell
    return betas, cells


def write_json_csv(betas, cells):
    out = {
        "_meta": build_meta("analysis/beta_scan_report.py", WANDB_GROUPS, study_root=STUDY),
        "metric_note": (
            "openness = total_openness = sum(1 - sigma) per family (nats; ~0 = closed). "
            "n_eff_open_frac deliberately NOT headlined (scale-invariant, misleads when closed)."
        ),
        "wandb_project_url": WANDB_PROJECT_URL,
        "mults": MULTS,
        "betas": betas,
        "families": {fam: n for fam, _l, n in FAMILIES},
        "cells": cells,
    }
    (HERE / "beta_scan_summary.json").write_text(json.dumps(out, indent=2))
    # flat CSV
    lines = ["family,N,mult,beta,openness_mean,openness_sem,frac_half_open_mean,heldout_ll_mean,heldout_ll_sem,n"]
    for m in MULTS:
        for b in betas:
            key = f"m{m}_b{b:g}"
            c = cells.get(key)
            if not c:
                continue
            for fam, _l, N in FAMILIES:
                o = c[f"{fam}_openness"]; h = c["heldout_ll"]; fh = c[f"{fam}_frac_half_open"]
                lines.append(
                    f"{fam},{N},{m},{b:g},{o['mean']:.5f},{o['sem']:.5f},"
                    f"{fh['mean']:.5f},{h['mean']:.5f},{h['sem']:.5f},{o['n']}"
                )
    (HERE / "beta_scan_summary.csv").write_text("\n".join(lines) + "\n")


def _bcolors(betas):
    return {b: c for b, c in zip(betas, plt.cm.viridis(np.linspace(0.15, 0.82, len(betas))))}


def _raw(rows, m, b, col):
    """Per-run values for a (mult, beta) cell and a grid CSV column."""
    return [_f(r[col]) for r in rows
            if _f(r["mult"]) == m and _f(r["beta"]) == b and _f(r[col]) is not None]


# horizontal offset (in x-index units) applied per beta so each beta's raw dots
# sit just to the right of its shrunk mean marker, with a little jitter.
_RAW_DX = 0.10
_RAW_JIT = 0.025


def _overlay_raw(ax, rows, betas, bc, xpos, col):
    rng = np.random.default_rng(0)
    for i, b in enumerate(betas):
        dx = _RAW_DX + i * 0.0  # same side for all betas; color disambiguates
        for m in MULTS:
            vals = _raw(rows, m, b, col)
            if not vals:
                continue
            xj = xpos[m] + dx + rng.uniform(-_RAW_JIT, _RAW_JIT, size=len(vals))
            ax.scatter(xj, vals, s=13, color=bc[b], alpha=0.35, edgecolors="none", zorder=1)


def fig_bottlenecks_by_mult(betas, cells, rows):
    """6-family openness vs multiplier, one panel per family, lines by base beta."""
    apply_presentation_style()
    bc = _bcolors(betas)
    xpos = {m: i for i, m in enumerate(MULTS)}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, (fam, lbl, N) in zip(axes.ravel(), FAMILIES):
        _overlay_raw(ax, rows, betas, bc, xpos, f"{fam}.total_openness")
        for b in betas:
            xs, ms, es = [], [], []
            for m in MULTS:
                c = cells.get(f"m{m}_b{b:g}")
                if not c or c[f"{fam}_openness"]["mean"] is None:
                    continue
                xs.append(xpos[m]); ms.append(c[f"{fam}_openness"]["mean"]); es.append(c[f"{fam}_openness"]["sem"])
            ax.errorbar(xs, ms, yerr=es, fmt="o-", ms=5, color=bc[b], mfc="white", mec=bc[b],
                        ecolor=bc[b], elinewidth=1.4, capsize=3, lw=2.0, zorder=3, label=f"\u03b2={b:g}")
        tag = "  \u2190 multiplier target" if fam == "update_net_latent" else ""
        ax.set_title(f"{lbl}{tag}  (N={N})", fontsize=13)
        ax.set_xticks(list(xpos.values())); ax.set_xticklabels(MULTS)
        ax.set_ylim(bottom=-0.1)
        ax.axhline(0.05, color="0.8", ls=":", lw=1.0)
    for ax in axes[1, :]:
        ax.set_xlabel("update-net latent penalty multiplier")
    for ax in axes[:, 0]:
        ax.set_ylabel("total openness  \u03a3(1\u2212\u03c3)")
    axes[0, 2].legend(title="base \u03b2", frameon=False, loc="upper right")
    fig.suptitle("Bottleneck openness vs multiplier, by base \u03b2 (mean\u00b1SEM over seeds/lr)",
                 fontsize=15, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.02, 0.005,
             "Markers: mean \u00b1 SEM. Faded dots: individual runs (jittered, offset right of the mean). "
             "n = 3\u20137 per point (seeds \u00d7 lr; mult=10 pools the mult10-supp launch). "
             "openness = \u03a3(1\u2212\u03c3); n_eff_open_frac deliberately not used.",
             fontsize=9, color="0.35", ha="left")
    fig.savefig(HERE / "fig_bottlenecks_by_mult.png", bbox_inches="tight")
    plt.close(fig)


def fig_mult_axis_heldout(betas, cells, rows):
    """Two-panel: update-net-latent openness (top) + held-out LL (bottom) vs multiplier."""
    apply_presentation_style()
    bc = _bcolors(betas)
    xpos = {m: i for i, m in enumerate(MULTS)}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True)
    _overlay_raw(ax1, rows, betas, bc, xpos, "update_net_latent.total_openness")
    _overlay_raw(ax2, rows, betas, bc, xpos, "heldout_eval_ll")
    for b in betas:
        for ax, key in [(ax1, "update_net_latent_openness"), (ax2, "heldout_ll")]:
            xs, ms, es = [], [], []
            for m in MULTS:
                c = cells.get(f"m{m}_b{b:g}")
                if not c or c[key]["mean"] is None:
                    continue
                xs.append(xpos[m]); ms.append(c[key]["mean"]); es.append(c[key]["sem"])
            ax.errorbar(xs, ms, yerr=es, fmt="o-", ms=5, color=bc[b], mfc="white", mec=bc[b],
                        ecolor=bc[b], elinewidth=1.4, capsize=3, lw=2.0, zorder=3, label=f"\u03b2={b:g}")
    ax1.set_ylabel("interaction openness\n\u03a3(1\u2212\u03c3)  (update\u2190latent)")
    ax1.axhline(0.05, color="0.8", ls=":", lw=1.0)
    ax2.set_ylabel("held-out mouse\ntransfer likelihood")
    ax2.set_xticks(list(xpos.values())); ax2.set_xticklabels([f"mult={m}" for m in MULTS])
    ax2.set_xlabel("update-net latent penalty multiplier")
    ax1.legend(title="base \u03b2", frameon=False, loc="upper right")
    fig.suptitle("Multiplier monotonically closes the interaction bottleneck;\nheld-out transfer is flat (set by \u03b2, not multiplier)",
                 fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.02, 0.005,
             "Markers: mean \u00b1 SEM. Faded dots: individual runs (jittered, offset right of the mean). "
             "n = 3\u20137 per point (seeds \u00d7 lr; mult=10 pools the mult10-supp launch).",
             fontsize=9, color="0.35", ha="left")
    fig.savefig(HERE / "fig_mult_axis_heldout.png", bbox_inches="tight")
    plt.close(fig)


def main():
    rows = _read_grid()
    betas, cells = summarize(rows)
    write_json_csv(betas, cells)
    fig_bottlenecks_by_mult(betas, cells, rows)
    fig_mult_axis_heldout(betas, cells, rows)
    import update_reports
    data = json.loads((HERE / "beta_scan_summary.json").read_text())
    updated = update_reports.run(data)
    print(f"beta_scan_report: {len(cells)} cells, {sum(1 for r in rows)} runs; reports updated: {updated}")


if __name__ == "__main__":
    main()
