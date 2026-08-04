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
import numpy as np

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
D_NOMINAL = (10, 30, 100, 300, 614)   # the cohort sizes the sweep asked for
D_SNAP_TOL = 3                        # resolved D lands within a mouse or two of nominal

# The grid's winning operating point -- focal series in the verdict figure.
FOCAL = (1, 0.0003)                   # (mult, beta)
S05_FIXED = (2, 0.001)                # study 05's fixed penalty -- the curve being corrected
FOCAL_COLOR = "#1f77b4"
S05_COLOR = "#c44e52"                 # study 05's fixed penalty, the curve being corrected
CONTEXT_GREY = "#b0b0b0"              # the other seven penalty settings
# Best per-mouse classical RL baseline (compare-to-threshold), study 05 r1 -- the bar to clear.
RL_BASELINE = 0.7170


def _fmt_beta(b: float) -> str:
    """Consistent beta formatting across every figure/label ('3e-4', not '0.0003' or '0.0003000')."""
    return f"{b:.0e}".replace("e-0", "e-")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_own_grid():
    """Rows usable for analysis = a trustworthy FINAL heldout_ll.

    Normally that means state=='finished'. It also includes runs whose scalar was recovered
    post-hoc by backfill_lost_heldout.py: those completed training AND their held-out stage
    (Beaker exit 0, committed output+table artifacts) but lost their final summary write to a
    W&B heartbeat timeout, so they sit at state=='crashed' with an exact, table-derived metric.
    Excluding them would silently drop 5 D=300 cells. Any OTHER crashed run is still excluded --
    its heldout_ll, if any, is a mid-training incremental value, not a final one.
    """
    with GRID_CSV.open() as f:
        rows = list(csv.DictReader(f))
    usable = [r for r in rows
              if _f(r["heldout_ll"]) is not None
              and (r["state"] == "finished" or r.get("heldout_backfilled") == "True")]
    return rows, usable


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


def nominal_d(d: int) -> int:
    """Snap a run's ACTUAL resolved D to the grid's nominal cohort size.

    The sweep requests a *ratio* of the 614-mouse cohort, so the resolved subject count
    lands a mouse or two either side of the nominal target (29/30, 99/101, 300/301 all
    appear in grid.csv). Grouping on the raw value silently splits every cell in half and
    reports n=1 with sem=0.0000 for what are really 2-seed cells -- so bucket first.
    Tolerance is +/-3 mice; anything further from a nominal D is a genuinely different
    cohort and is left alone (and will show up as its own row, which is the loud failure
    we want rather than a silent mis-merge).
    """
    for target in D_NOMINAL:
        if abs(d - target) <= D_SNAP_TOL:
            return target
    return d


def summarize(finished):
    cells = {}
    for r in finished:
        d, mult, beta = int(float(r["D"])), int(float(r["mult"])), float(r["beta"])
        key = (nominal_d(d), mult, beta)
        cells.setdefault(key, []).append(_f(r["heldout_ll"]))
    out = []
    for (d, mult, beta), vals in sorted(cells.items()):
        mean = sum(vals) / len(vals)
        sem = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / max(len(vals) - 1, 1) ** 0.5 if len(vals) > 1 else 0.0
        out.append({"D": d, "mult": mult, "beta": beta, "n": len(vals),
                    "heldout_ll_mean": round(mean, 4), "heldout_ll_sem": round(sem, 4)})
    return out


def fig_scaling_surface(cells, s05_curve, n_usable):
    """The verdict figure: the tuned operating point rises with D; the fixed one doesn't.

    Deliberately NOT an 8-way equal-weight spaghetti plot. The grid's scientific content is
    one contrast -- the lightest-total-penalty setting (mult=1, beta=3e-4) is one of only TWO
    of the 8 settings (the other is mult=10, beta=3e-4) with no step-to-step drop exceeding its
    own SEM anywhere across D. Of the other six: three have exactly one SEM-clearing dip apiece
    at varying points in D (not all "mid-range", not all recovering); three (including study
    05's own fixed point, mult=2 beta=1e-3) have a small negative step that does NOT clear its
    SEM, i.e. are flat within noise rather than genuinely declining. So "declines" describes a
    minority of the seven non-focal settings -- see r2's point 2 and fig_beta_mult_sensitivity
    for the full per-series breakdown; don't restate a blanket "the rest decline" claim here.
    The winner is drawn focal (saturated, heavy, direct-labelled) and the other seven as a
    low-weight context band showing the spread. Reference levels: the GRU ceiling (study 01)
    and the best per-mouse classical RL baseline (study 05 r1), the bar this study had to clear.
    """
    by_series = {}
    for c in cells:
        by_series.setdefault((c["mult"], c["beta"]), []).append(c)
    for k in by_series:
        by_series[k].sort(key=lambda p: p["D"])
    if FOCAL not in by_series:                     # partial grid -- fall back to no focal series
        focal_pts = None
    else:
        focal_pts = by_series[FOCAL]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    # --- reference levels -------------------------------------------------------------
    ds = sorted(GRU)
    ax.plot(ds, [GRU[d] for d in ds], "--", color="#333333", lw=1.6, zorder=4)
    ax.plot(ds, [GRU[d] for d in ds], "o", color="#333333", ms=4.5, zorder=4)
    ax.axhline(RL_BASELINE, color="#b8860b", lw=1.1, ls=(0, (5, 3)), zorder=1)

    # --- context: the seven non-winning penalty settings ------------------------------
    for key, pts in by_series.items():
        if key == FOCAL:
            continue
        ax.errorbar([p["D"] for p in pts], [p["heldout_ll_mean"] for p in pts],
                    yerr=[p["heldout_ll_sem"] for p in pts],
                    fmt="-", marker="o", ms=3, color=CONTEXT_GREY, lw=0.9, alpha=0.55,
                    capsize=0, zorder=2)

    # --- study 05's fixed penalty: the curve this study corrects ----------------------
    if s05_curve:
        dsx = sorted(s05_curve)
        ax.plot(dsx, [s05_curve[d] for d in dsx], ":", color=S05_COLOR, lw=1.8, zorder=3)
        ax.plot(dsx, [s05_curve[d] for d in dsx], "D", color=S05_COLOR, ms=4.5, zorder=3)

    # --- focal: the winning operating point -------------------------------------------
    if focal_pts:
        ax.errorbar([p["D"] for p in focal_pts], [p["heldout_ll_mean"] for p in focal_pts],
                    yerr=[p["heldout_ll_sem"] for p in focal_pts],
                    fmt="-", marker="o", ms=7, color=FOCAL_COLOR, lw=2.4, capsize=3,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlim(8, 1500)                      # right headroom for the direct labels
    ax.set_xlabel("training mice  $D$  (log scale)")
    ax.set_ylabel("held-out likelihood  (unseen mice)")

    # --- direct labels in the right-hand whitespace (figure-style 6.3 / 7.3) ----------
    # Anchor each label at its series' final value, then push overlapping labels apart
    # vertically (they are dense between 0.715 and 0.717) and draw a leader back to the
    # series so every label still resolves to the row it names (figure-style 6.9).
    wants = [(GRU[614], "GRU\n(study 01)", "#333333", "normal", 2)]
    if focal_pts:
        wants.append((focal_pts[-1]["heldout_ll_mean"],
                      "disRNN, tuned\nmult=1, β=3e-4", FOCAL_COLOR, "bold", 2))
    if s05_curve:
        wants.append((s05_curve[max(s05_curve)],
                      "disRNN, fixed penalty\n(study 05: mult=2, β=1e-3)", S05_COLOR, "normal", 2))
    wants.append((RL_BASELINE, "best per-mouse\nRL baseline", "#b8860b", "normal", 2))

    lo, hi = ax.get_ylim()
    span = hi - lo
    wants.sort(key=lambda w: w[0])
    placed, min_gap = [], span * 0.105          # ~2 text lines at this font size
    for anchor, text, color, weight, nlines in wants:
        y = anchor
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, anchor, text, color, weight))
    # keep the stack inside the axes
    overshoot = placed[-1][0] - (hi - span * 0.05)
    if overshoot > 0:
        placed = [(y - overshoot, a, t, c, w) for y, a, t, c, w in placed]

    for y, anchor, text, color, weight in placed:
        ax.annotate(text, xy=(645, anchor), xytext=(760, y), color=color, fontsize=8,
                    weight=weight, va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.55,
                                    shrinkA=1, shrinkB=1))

    ax.annotate("7 heavier-penalty\nsettings", xy=(300, 0.7150), xytext=(120, 0.7040),
                color="#6e6e6e", fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.8,
                                shrinkA=0, shrinkB=3))

    ax.set_title("Only the lightest penalty (mult=1, β=3e-4) rises across the whole cohort range\n"
                 "(higher = better)", loc="left", fontsize=10.5)
    ax.text(0.985, 0.03, f"n = 2 seeds per point, error bars SEM · {n_usable}/{N_TOTAL} runs",
            transform=ax.transAxes, fontsize=7.5, color="#6e6e6e", ha="right")
    ax.grid(alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.10, right=0.775, top=0.90, bottom=0.11)
    out = HERE / "fig_scaling_surface.png"
    fig.savefig(out, dpi=200)
    return out


def fig_beta_mult_sensitivity(cells):
    """Companion to fig_scaling_surface: which (mult, beta) wins at each D, and by how much.

    The scaling-surface figure only shows the winning series against reference curves; it
    can't show the crossover (beta=3e-4 is WORST at D=10, BEST from D=100 on) or how much the
    operating-point choice matters at each cohort size. Skipped when the grid isn't the full
    8-setting design (partial early runs) since the heatmap assumes a complete D x mult x beta
    rectangle.
    """
    mults = sorted({c["mult"] for c in cells})
    betas = sorted({c["beta"] for c in cells})
    ds = sorted({c["D"] for c in cells})
    by = {(c["mult"], c["beta"], c["D"]): c["heldout_ll_mean"] for c in cells}
    row_keys = [(m, b) for b in betas for m in mults]
    if any((m, b, d) not in by for m, b in row_keys for d in ds):
        return None  # incomplete grid -- don't render a heatmap with holes

    mat = np.array([[by[(m, b, d)] for d in ds] for m, b in row_keys])
    vmin, vmax = mat.min(), mat.max()

    fig = plt.figure(figsize=(9.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[8, 2.6], width_ratios=[1, 0.045],
                          hspace=0.55, wspace=0.05)
    ax, cax, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])

    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            col_best = np.isclose(v, mat[:, j].max())
            color = "white" if (v - vmin) / (vmax - vmin) < 0.55 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.2,
                    color=color, weight="bold" if col_best else "normal")
            if col_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="white", lw=2.2))

    ax.set_xticks(range(len(ds))); ax.set_xticklabels([str(d) for d in ds])
    ax.set_yticks(range(len(row_keys))); ax.set_yticklabels([f"mult={m}" for m, b in row_keys], fontsize=8.5)
    ax.set_xlabel("training mice  D")
    n_per_beta = len(mults)
    for line in range(n_per_beta, len(row_keys), n_per_beta):
        ax.axhline(line - 0.5, color="white", lw=2.5)
    for grp, b in enumerate(betas):
        ax.text(-1.35, grp * n_per_beta + (n_per_beta - 1) / 2, f"β={_fmt_beta(b)}",
                rotation=90, va="center", ha="center", fontsize=9.5, weight="bold")
    ax.set_title("Which penalty wins depends on cohort size — "
                 "the light penalty overtakes past D≈100", loc="left", fontsize=10.5)
    fig.colorbar(im, cax=cax, label="held-out\nlikelihood")
    cax.tick_params(labelsize=7.5)

    spread = mat.max(axis=0) - mat.min(axis=0)
    axb.bar(range(len(ds)), spread, color="#6e6e6e", width=0.55)
    for j, s in enumerate(spread):
        axb.text(j, s + spread.max() * 0.03, f"{s:.4f}", ha="center", va="bottom",
                 fontsize=7.5, color="#444")
    axb.set_xticks(range(len(ds))); axb.set_xticklabels([str(d) for d in ds])
    axb.set_xlabel("training mice  D")
    axb.set_ylabel("range across\n8 settings", fontsize=8)
    axb.set_title("Operating-point choice matters most at the smallest and largest cohorts",
                  loc="left", fontsize=9.5)
    axb.spines[["top", "right"]].set_visible(False)
    axb.set_ylim(0, spread.max() * 1.35)
    for a in (ax, axb):
        a.tick_params(axis="both", labelsize=8.5)

    fig.subplots_adjust(left=0.16, right=0.87)
    out = HERE / "fig_beta_mult_sensitivity.png"
    fig.savefig(out, dpi=200)
    return out


def summarize_gap(finished):
    """Per-(D,mult,beta) mean generalization gap = in-sample (eval_ll) - held-out (heldout_ll).

    gap > 0 = the model fits the training cohort better than it transfers (overfits).
    Mirrors summarize() exactly but on the gap instead of heldout_ll -- same D-bucketing,
    same n/seed bookkeeping.
    """
    cells = {}
    for r in finished:
        ev = _f(r["eval_ll"])
        if ev is None:
            continue
        d, mult, beta = nominal_d(int(float(r["D"]))), int(float(r["mult"])), float(r["beta"])
        cells.setdefault((d, mult, beta), []).append(ev - _f(r["heldout_ll"]))
    out = []
    for (d, mult, beta), vals in sorted(cells.items()):
        mean = sum(vals) / len(vals)
        sem = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / max(len(vals) - 1, 1) ** 0.5 if len(vals) > 1 else 0.0
        out.append({"D": d, "mult": mult, "beta": beta, "n": len(vals),
                    "gap_mean": round(mean, 4), "gap_sem": round(sem, 4)})
    return out


def fig_generalization_gap(gap_cells):
    """The overfitting-vs-D companion to fig_scaling_surface: in-sample minus held-out likelihood.

    Every one of the 8 (mult,beta) settings shows the SAME qualitative shape -- gap falls
    steeply from D=10 to a minimum near D=100, then rises again by D=300, then is flat within
    seed noise from D=300 to D=614 (NOT still rising -- the 300->614 step is smaller than the
    per-seed spread at every one of the 8 cells; see r2 report for the check). So this is one
    story about D, not a story that separates the settings much -- drawn with all 8 series at
    low weight except the two reference points (focal winner + study 05's fixed penalty),
    which track each other closely throughout.
    """
    mults = sorted({c["mult"] for c in gap_cells})
    betas = sorted({c["beta"] for c in gap_cells})
    ds = sorted({c["D"] for c in gap_cells})
    by = {(c["mult"], c["beta"], c["D"]): c["gap_mean"] for c in gap_cells}
    if any((m, b, d) not in by for m in mults for b in betas for d in ds):
        return None  # incomplete grid

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axhline(0, color="#333333", lw=1.1)
    for m in mults:
        for b in betas:
            if (m, b) in (FOCAL, S05_FIXED):
                continue
            ax.plot(ds, [by[(m, b, d)] for d in ds], "-o", ms=3, color=CONTEXT_GREY,
                    lw=0.9, alpha=0.55, zorder=2)
    ax.plot(ds, [by[(FOCAL[0], FOCAL[1], d)] for d in ds], "-o", ms=7, color=FOCAL_COLOR,
            lw=2.4, zorder=4, markeredgecolor="white", markeredgewidth=0.8,
            label=f"disRNN, tuned (mult={FOCAL[0]}, β={_fmt_beta(FOCAL[1])})")
    ax.plot(ds, [by[(S05_FIXED[0], S05_FIXED[1], d)] for d in ds], ":D", ms=5, color=S05_COLOR,
            lw=1.8, zorder=3,
            label=f"disRNN, fixed penalty (study 05: mult={S05_FIXED[0]}, β={_fmt_beta(S05_FIXED[1])})")

    ax.set_xscale("log")
    ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("training mice  D  (log scale)")
    ax.set_ylabel("generalization gap\n(in-sample − held-out likelihood)")
    ax.set_title("Overfitting dips near D=100, then plateaus from D=300 to D=614",
                 loc="left", fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.text(0.0, 1.06, "gap > 0 = overfits the training cohort", transform=ax.transAxes,
            fontsize=8, color="#6e6e6e")
    ax.text(0.985, 0.04, "n = 2 seeds per point, mean shown; other 6 settings in grey",
            transform=ax.transAxes, fontsize=7.5, color="#6e6e6e", ha="right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    out = HERE / "fig_generalization_gap.png"
    fig.savefig(out, dpi=200)
    return out


def update_report_block(cells, n_usable, n_running, n_outstanding, n_failed):
    lines = ["| D | mult | β | held-out (mean) | sem | n seeds |",
             "|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['D']} | {c['mult']} | {c['beta']:g} | "
                     f"{c['heldout_ll_mean']:.4f} | {c['heldout_ll_sem']:.4f} | {c['n']} |")
    status_line = (f"**Progress: {n_usable}/{N_TOTAL} usable, {n_running} running, "
                   f"{n_outstanding} outstanding, {n_failed} failed W&B runs.**")
    block = ("<!-- BEGIN result-1 -->\n" + status_line + "\n\n" + "\n".join(lines)
             + "\n<!-- END result-1 -->")
    if not REPORT.exists():
        return
    text = REPORT.read_text()
    new = re.sub(r"<!-- BEGIN result-1 -->.*?<!-- END result-1 -->", block, text, flags=re.S)
    REPORT.write_text(new)


def main() -> None:
    rows, usable = read_own_grid()
    s05_curve = read_s05_fixed_curve()
    cells = summarize(usable)

    # W&B's own state field cleanly distinguishes preemption ("crashed", benign -- autoResume
    # reuses the same run id and keeps training) from a real script failure ("failed", e.g. a
    # NaN ValueError) -- no need to re-derive this from Beaker job history here.
    # n_usable is the headline: state=='finished' PLUS backfilled-but-crashed (see read_own_grid).
    n_usable = len(usable)
    n_backfilled = sum(1 for r in usable if r.get("heldout_backfilled") == "True")
    n_finished = sum(1 for r in rows if r["state"] == "finished")
    n_running = sum(1 for r in rows if r["state"] == "running")
    n_crashed = sum(1 for r in rows if r["state"] == "crashed")
    n_failed = sum(1 for r in rows if r["state"] == "failed")
    # Grid points still lacking a trustworthy final value. NOT N_TOTAL - len(rows): rescues,
    # NaN-retries and the tier-1 relaunch each opened a FRESH W&B run id for a grid point that
    # already had a (failed) run, so len(rows) exceeds N_TOTAL and that subtraction goes
    # negative (the committed r2 block read "-10 pending"). Count uncovered cells instead.
    n_outstanding = max(N_TOTAL - n_usable, 0)

    fig_path = fig_scaling_surface(cells, s05_curve, n_usable)
    sens_path = fig_beta_mult_sensitivity(cells)
    gap_cells = summarize_gap(usable)
    gap_fig_path = fig_generalization_gap(gap_cells)

    payload = {"_meta": build_meta("analysis/scaling_report.py", WANDB_GROUPS, study_root=STUDY),
               "note": ("LIVE report -- regenerate as the grid progresses. heldout_ll only "
                        "trusted for state=='finished' rows (written incrementally otherwise), "
                        "PLUS backfilled runs whose exact scalar was recovered post-hoc from the "
                        "per-subject table (backfill_lost_heldout.py). 'crashed' = preempted, "
                        "autoResume reuses the run id; 'failed' = a real script error (e.g. NaN)."),
               "progress": {"usable": n_usable, "of_which_backfilled": n_backfilled,
                            "finished": n_finished, "running": n_running, "crashed": n_crashed,
                            "failed": n_failed, "outstanding": n_outstanding,
                            "n_wandb_runs": len(rows), "n_total": N_TOTAL},
               "cells": cells, "generalization_gap_cells": gap_cells}
    (HERE / "summary.json").write_text(json.dumps(payload, indent=2))
    update_report_block(cells, n_usable, n_running, n_outstanding, n_failed)
    fig_names = ", ".join(p.name for p in (fig_path, sens_path, gap_fig_path) if p)
    print(f"wrote {fig_names} and summary.json  ({n_usable}/{N_TOTAL} usable "
          f"({n_backfilled} backfilled), {n_finished} wandb-finished, "
          f"{len(cells)} (D,mult,beta) cells with >=1 seed)")


if __name__ == "__main__":
    main()
