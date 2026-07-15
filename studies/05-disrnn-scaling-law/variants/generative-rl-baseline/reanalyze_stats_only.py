#!/usr/bin/env python
"""Recompute switch + history-dependent STATS from cached histories -- no plots, no re-simulation.

Two prior attempts both timed out:
  1. run_rl_rollout.py (6h wall clock): the switch curve finished; history-dependent
     significance testing (10x more pattern bins, each with a 250-resample bootstrap) did not.
  2. reanalyze_from_cache.py + DISRNN_SUBJECT_BOOTSTRAP_RESAMPLES=0 (2h wall clock): STILL timed
     out, which proved the bootstrap was never the real bottleneck.

Root cause, found by checking file mtimes against the 2h run: `run_post_training_analysis_from_
histories()` computes ALL stats first (both switch and history, done in well under 12 minutes --
the run's own log shows the first FIGURE appearing at +12 min from launch), then spends the
remaining ~1h50m on PLOTS -- in particular per-session scatter plots over 18,124 points, at
~20-25 minutes each. r4 never reads a single one of these figures; it only reads the point
estimates in `quantitative_summary` / `delta_significance_summary`.

So call the two STATS functions directly -- `compute_switch_stats()` and
`compute_history_dependent_switch_stats()` -- and skip `_save_switch_figures()` /
`_save_history_dependent_switch_figures()` entirely. Same wrapper code (nothing reimplemented),
same statistics, zero re-simulation, zero rendering.

Usage (CPU-only; submit via submit_reanalyze_stats_only.sbatch):
    python reanalyze_stats_only.py --alias ctt --dir <the run_rl_rollout.py --out dir for ctt>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ALIAS_SUBDIR = {
    "ctt": "ForagingCompareThreshold",
    "bari": "QLearning_L1F1_CK1_softmax",
    "hattori": "QLearning_L2F1_softmax",
}


def _to_serializable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True, choices=sorted(ALIAS_SUBDIR))
    ap.add_argument("--dir", required=True, help="the --out/<alias> dir from run_rl_rollout.py")
    ap.add_argument("--wrapper", default="/home/han.hou/code/aind-disrnn-wrapper/code")
    args = ap.parse_args()

    sys.path.insert(0, args.wrapper)
    from post_training_analysis.generative_analysis import (
        compute_history_dependent_switch_stats,
        compute_switch_stats,
    )

    d = Path(args.dir).expanduser().resolve()
    subdir = ALIAS_SUBDIR[args.alias]
    sim_path = d / "analysis" / subdir / "simulated_session_history.pkl"
    animal_path = d / "analysis" / "animal_session_history.pkl"

    print(f"=== {args.alias}: stats-only reanalysis from cache ({d}) ===", flush=True)
    for p in (sim_path, animal_path):
        if not p.exists():
            raise FileNotFoundError(f"missing cached input: {p}")

    t0 = time.perf_counter()
    with sim_path.open("rb") as f:
        simulated_sessions = pickle.load(f)
    with animal_path.open("rb") as f:
        animal_sessions = pickle.load(f)
    print(f"  loaded {len(animal_sessions)} animal / {len(simulated_sessions)} simulated "
          f"sessions in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    switch_stats = compute_switch_stats(animal_sessions=animal_sessions,
                                         simulated_sessions=simulated_sessions)
    print(f"  compute_switch_stats: {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    history_stats = compute_history_dependent_switch_stats(
        animal_sessions=animal_sessions, simulated_sessions=simulated_sessions
    )
    print(f"  compute_history_dependent_switch_stats: {time.perf_counter() - t0:.1f}s", flush=True)

    out_dir = d / "analysis" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "switch_stats_no_figures.json").write_text(
        json.dumps(_to_serializable(switch_stats), indent=2)
    )
    (out_dir / "history_dependent_switch_stats_no_figures.json").write_text(
        json.dumps(_to_serializable(history_stats), indent=2)
    )
    combined = {
        "switch_triggered": {
            "quantitative_summary": switch_stats.get("quantitative_summary", {}),
            "delta_significance_summary": switch_stats.get("delta_significance_summary", {}),
        },
        "history_dependent": {
            "quantitative_summary": history_stats.get("quantitative_summary", {}),
            "delta_significance_summary": history_stats.get("delta_significance_summary", {}),
        },
    }
    (d / "quantitative_summary.json").write_text(
        json.dumps(_to_serializable(combined), indent=2)
    )
    print(f"  done -> {d / 'quantitative_summary.json'} (no figures written)", flush=True)


if __name__ == "__main__":
    main()
