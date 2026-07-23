"""r2 producer: the mult-d-grid held-out-transfer surface, live while the grid is in flight.

Reads the committed analysis/grid.csv (offline; run `make pull` first to refresh from W&B) plus
study 05's committed grid.csv (for the fixed-penalty reference curve). Only `state == "finished"`
rows are trusted for heldout_ll -- it is written incrementally throughout training, so an
in-flight run's value is not final (see pull_grid.py docstring).

    python analysis/scaling_report.py          # offline; no WANDB_API_KEY needed

OUTPUTS (all committed so the report renders in-repo, and so each periodic debrief can attach
the freshest figure without needing a fresh W&B pull each time it's sent):
  analysis/summary.json           - curated per-(D,mult,beta) stats + _meta provenance
  analysis/fig_scaling_surface.png - held-out LL vs D, colour=beta, marker=mult, vs GRU + 05's
                                      fixed-penalty curve; title carries live progress N/80.
Regenerates the <!-- BEGIN result-1 --> block in reports/r2-scaling-surface.md.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402

GRID_CSV = HERE / "grid.csv"
S05_GRID_CSV = REPO / "studies/05-disrnn-scaling-law/analysis/grid.csv"
REPORT = HERE / "reports" / "r2-scaling-surface.md"
WANDB_GROUPS = ["mult-d-grid@20260718-151409"]

GRU = {10: 0.7219, 30: 0.7250, 100: 0.7262, 300: 0.7267, 614: 0.7268}  # study 01
BETA_COLOR = {0.0003: "#1f77b4", 0.001: "#d62728"}
MULT_MARKER = {1: "o", 2: "s", 5: "^", 10: "x"}
N_TOTAL = 80


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_own_grid():
    with GRID_CSV.open() as f:
        rows = list(csv.DictReader(f))
    finished = [r for r in rows if r["state"] == "finished" and _f(r["heldout_ll"]) is not None]
    return rows, finished


def read_s05_fixed_curve():
    """study 05 dscan-mult2: mult=2, beta=1e-3, the fixed-penalty curve this study corrects."""
    if not S05_GRID_CSV.exists():
        return {}
    with S05_GRID_CSV.open() as f:
        rows = list(csv.DictReader(f))
    by_d = {}
    for r in rows:
        if r["variant"] != "dscan-mult2" or r["state"] != "finished":
            continue
        d = int(float(r["D"]))
        ll = _f(r["heldout_ll"])
        if ll is not None:
            by_d.setdefault(d, []).append(ll)
    return {d: sum(v) / len(v) for d, v in by_d.items()}


def summarize(finished):
    cells = {}
    for r in finished:
        d, mult, beta = int(float(r["D"])), int(float(r["mult"])), float(r["beta"])
        key = (d, mult, beta)
        cells.setdefault(key, []).append(_f(r["heldout_ll"]))
    out = []
    for (d, mult, beta), vals in sorted(cells.items()):
        mean = sum(vals) / len(vals)
        sem = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / max(len(vals) - 1, 1) ** 0.5 if len(vals) > 1 else 0.0
        out.append({"D": d, "mult": mult, "beta": beta, "n": len(vals),
                    "heldout_ll_mean": round(mean, 4), "heldout_ll_sem": round(sem, 4)})
    return out


def fig_scaling_surface(cells, s05_curve, n_finished):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ds = sorted(GRU)
    ax.plot(ds, [GRU[d] for d in ds], "o--", color="#333333", ms=6, label="GRU (study 01)", zorder=2)
    if s05_curve:
        dsx = sorted(s05_curve)
        ax.plot(dsx, [s05_curve[d] for d in dsx], "d:", color="#888888", ms=6,
                label="disRNN fixed penalty (05, mult=2 β=1e-3)", zorder=2)

    by_series = {}
    for c in cells:
        by_series.setdefault((c["mult"], c["beta"]), []).append(c)
    for (mult, beta), pts in by_series.items():
        pts = sorted(pts, key=lambda p: p["D"])
        xs = [p["D"] for p in pts]
        ys = [p["heldout_ll_mean"] for p in pts]
        es = [p["heldout_ll_sem"] for p in pts]
        ax.errorbar(xs, ys, yerr=es, fmt=MULT_MARKER.get(mult, "*"),
                    color=BETA_COLOR.get(round(beta, 4), "k"), ms=8, capsize=3, lw=1.2,
                    label=f"mult={mult}, β={beta:g}", zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(sorted(GRU))
    ax.set_xticklabels([str(d) for d in sorted(GRU)])
    ax.set_xlabel("training mice  D  (log scale)")
    ax.set_ylabel("held-out likelihood")
    ax.set_title(f"mult-d-grid — live scaling surface  ({n_finished}/{N_TOTAL} finished)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = HERE / "fig_scaling_surface.png"
    fig.savefig(out, dpi=130)
    return out


def update_report_block(cells, n_finished, n_running, n_pending, n_failed):
    lines = ["| D | mult | β | held-out (mean) | sem | n seeds |",
             "|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['D']} | {c['mult']} | {c['beta']:g} | "
                     f"{c['heldout_ll_mean']:.4f} | {c['heldout_ll_sem']:.4f} | {c['n']} |")
    status_line = (f"**Progress: {n_finished}/{N_TOTAL} finished, {n_running} running, "
                   f"{n_pending} pending, {n_failed} failed.**")
    block = ("<!-- BEGIN result-1 -->\n" + status_line + "\n\n" + "\n".join(lines)
             + "\n<!-- END result-1 -->")
    if not REPORT.exists():
        return
    text = REPORT.read_text()
    new = re.sub(r"<!-- BEGIN result-1 -->.*?<!-- END result-1 -->", block, text, flags=re.S)
    REPORT.write_text(new)


def main() -> None:
    rows, finished = read_own_grid()
    s05_curve = read_s05_fixed_curve()
    cells = summarize(finished)

    # W&B's own state field cleanly distinguishes preemption ("crashed", benign -- autoResume
    # reuses the same run id and keeps training) from a real script failure ("failed", e.g. a
    # NaN ValueError) -- no need to re-derive this from Beaker job history here.
    n_finished = sum(1 for r in rows if r["state"] == "finished")
    n_running = sum(1 for r in rows if r["state"] == "running")
    n_crashed = sum(1 for r in rows if r["state"] == "crashed")
    n_failed = sum(1 for r in rows if r["state"] == "failed")
    n_never_started = N_TOTAL - len(rows)  # tasks with no W&B run yet (still queued)

    fig_path = fig_scaling_surface(cells, s05_curve, n_finished)

    payload = {"_meta": build_meta("analysis/scaling_report.py", WANDB_GROUPS, study_root=STUDY),
               "note": ("LIVE report -- regenerate as the grid progresses. heldout_ll only "
                        "trusted for state=='finished' rows (written incrementally otherwise). "
                        "'crashed' = preempted, autoResume reuses the run id; 'failed' = a real "
                        "script error (e.g. NaN divergence)."),
               "progress": {"finished": n_finished, "running": n_running, "crashed": n_crashed,
                            "failed": n_failed, "never_started": n_never_started, "n_total": N_TOTAL},
               "cells": cells}
    (HERE / "summary.json").write_text(json.dumps(payload, indent=2))
    update_report_block(cells, n_finished, n_running, n_never_started, n_failed)
    print(f"wrote {fig_path.name} and summary.json  ({n_finished}/{N_TOTAL} finished, "
          f"{len(cells)} (D,mult,beta) cells with >=1 seed)")


if __name__ == "__main__":
    main()
