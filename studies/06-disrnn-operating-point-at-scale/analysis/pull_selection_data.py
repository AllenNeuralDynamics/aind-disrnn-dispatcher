#!/usr/bin/env python3
"""Pull the lr tag for study 03's D=100 runs and commit a clean lr=1e-3 selection CSV.

WHY THIS EXISTS. The committed study-03 grid (beta_scan_final_grid.csv) carries a MISLABELED `lr`
column (constant 0.01) — it mixes the two learning rates study 03 actually swept, lr∈{1e-3, 5e-3}.
For the penalty-selection plot both panels must be lr=1e-3 (matching study 05's D=614 panel and the
study-06 operating point; lr=5e-3 was NaN-prone in study 03). The LL *values* in the committed grid
are the source of truth; only the true `lr` tag is missing, so this reads it back from W&B (metadata
of EXISTING runs — no training) and writes a committed, lr-filtered CSV that model_selection.py then
reads OFFLINE (study-05 pull_grid.py convention).

JOIN. beta_scan_final_grid.csv `run` = last 8 hex of the W&B run.id; we map last8(id) -> config lr.

Needs WANDB_API_KEY (public internet, no VPN). Run once; commit analysis/data/d100_selection.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import wandb

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "studies/03-disrnn-beta-scan/analysis/beta_scan_final_grid.csv"
OUT = HERE / "data" / "d100_selection.csv"
PROJECT = "AIND-disRNN/disrnn_updnet_bottleneck_ratio_100mice"


def _lr_of(run) -> float | None:
    m = run.config.get("model")
    if isinstance(m, dict):
        lr = m.get("training", {}).get("lr")
        if lr is not None:
            return float(lr)
    lr = run.config.get("model.training.lr")
    return float(lr) if lr is not None else None


def main() -> None:
    os.environ.setdefault("WANDB_SILENT", "true")
    api = wandb.Api(timeout=30)
    id2lr = {}
    for r in api.runs(PROJECT, per_page=200):
        lr = _lr_of(r)
        if lr is not None:
            id2lr[r.id[-8:]] = lr          # last 8 hex == the grid's `run` key

    g = pd.read_csv(SRC).rename(columns={"in_eval_ll": "in_ll", "heldout_eval_ll": "heldout_ll"})
    g["run"] = g["run"].astype(str)
    g["true_lr"] = g["run"].map(id2lr)
    missing = g["true_lr"].isna().sum()
    if missing:
        print(f"WARNING: {missing}/{len(g)} runs had no lr from W&B (dropped)")
    g = g.dropna(subset=["true_lr", "in_ll", "heldout_ll"])
    g = g[g["true_lr"].round(4) == 0.001]  # lr=1e-3 only (match D=614 / study-06 operating point)
    g["D"] = 100
    out = g[["D", "beta", "mult", "seed", "in_ll", "heldout_ll"]].sort_values(["beta", "mult", "seed"])
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out)} lr=1e-3 runs)")
    print(out.groupby(["beta", "mult"]).size().to_string())


if __name__ == "__main__":
    main()
