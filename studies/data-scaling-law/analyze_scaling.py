#!/usr/bin/env python
"""Build the data-scaling curve: held-out-mouse generalization vs # training mice.

Pulls the study's runs from W&B (project mice_data_scaling), uses the ACTUAL number
of training subjects (len(resolved_subject_ids)) as x, the held-out generalization
likelihood as y, averages over seeds, fits L = E + (Dc/D)^alpha, and writes a CSV
(always) plus a PNG (if matplotlib is available).

Usage:
  .venv/bin/python studies/data-scaling-law/analyze_scaling.py \
      [--project AIND-disRNN/mice_data_scaling] [--out studies/data-scaling-law]
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import wandb

# y-axis: held-out-mouse generalization (mice never in training). Fall back to the
# logged history key if the summary wasn't populated (e.g. after a resume).
Y_SUMMARY_KEYS = ["heldout/test_likelihood", "heldout/eval_likelihood"]
Y_HISTORY_KEYS = ["heldout/eval_likelihood", "checkpoint/heldout_test_likelihood"]


def _subject_count(run) -> int | None:
    rs = run.config.get("resolved_subject_ids")
    return len(rs) if isinstance(rs, list) else None


def _heldout_value(run) -> float | None:
    for k in Y_SUMMARY_KEYS:
        v = run.summary.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # Fall back to the last logged history value (authoritative artifacts aside).
    for k in Y_HISTORY_KEYS:
        vals = [r[k] for r in run.history(keys=[k], pandas=False) if r.get(k) is not None]
        if vals:
            return float(vals[-1])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="AIND-disRNN/mice_data_scaling")
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()

    api = wandb.Api()
    rows = []
    for run in api.runs(args.project):
        if run.state != "finished":
            continue
        d = _subject_count(run)
        y = _heldout_value(run)
        if d is None or y is None:
            continue
        rows.append({"run": run.id, "D": d, "seed": run.config.get("seed"),
                     "heldout_likelihood": y})

    rows.sort(key=lambda r: (r["D"], r["seed"]))
    out = Path(args.out)
    csv_path = out / "scaling_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "D", "seed", "heldout_likelihood"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} runs)")

    # Aggregate over seeds per D.
    by_d: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        by_d[r["D"]].append(r["heldout_likelihood"])
    Ds = sorted(by_d)
    means = [sum(by_d[d]) / len(by_d[d]) for d in Ds]
    for d, m in zip(Ds, means):
        print(f"  D={d:5d}  heldout_LL_mean={m:.4f}  (n={len(by_d[d])})")

    # Power-law fit L = E + (Dc/D)^alpha via scipy if available (optional).
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        def model(D, E, Dc, alpha):
            return E + (Dc / D) ** alpha

        popt, _ = curve_fit(model, np.array(Ds, float), np.array(means, float),
                            p0=[max(means), Ds[0], 0.5], maxfev=20000)
        print(f"  fit: L = {popt[0]:.4f} + ({popt[1]:.2f}/D)^{popt[2]:.3f}")
    except Exception as e:  # scipy missing or fit failed — CSV still written
        popt = None
        print(f"  (power-law fit skipped: {e})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        for r in rows:
            ax.scatter(r["D"], r["heldout_likelihood"], c="0.7", s=18, zorder=2)
        ax.plot(Ds, means, "o-", color="C0", zorder=3, label="mean over seeds")
        ax.set_xscale("log")
        ax.set_xlabel("# training mice (D)")
        ax.set_ylabel("held-out-mouse likelihood")
        ax.set_title("Data-scaling: generalization vs training mice")
        ax.legend()
        png = out / "scaling_curve.png"
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
