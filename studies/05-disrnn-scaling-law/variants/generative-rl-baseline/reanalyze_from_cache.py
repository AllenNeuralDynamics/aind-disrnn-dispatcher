#!/usr/bin/env python
"""Recompute switch + history-dependent stats from the ALREADY-SIMULATED session histories.

The first `run_rl_rollout.py` pass for all three baselines hit the 6h wall clock: the switch curve
(what r4's headline needs) finished in every case, but the history-dependent significance testing
-- 10x more pattern bins than the switch curve, each with a 250-resample subject-level bootstrap --
did not. Re-simulating from scratch would waste the ~4-6h of RL rollout that already succeeded.

It does not need to. `run_rl_rollout.py` already wrote `simulated_session_history.pkl` and
`animal_session_history.pkl` to disk before the timeout, and the wrapper has a supported entrypoint
for exactly this case: `run_post_training_analysis_from_histories()` recomputes every stat --
switch AND history-dependent -- from saved histories, no model load, no re-simulation.

The bootstrap is what timed out, and it is decorative for r4's purposes: r4 reads a POINT ESTIMATE
(correlation, RMSE) from `quantitative_summary` / `delta_significance_summary`, which do not depend
on the bootstrap draws. Only the subject-level CONFIDENCE INTERVALS depend on it. So it is disabled
via `DISRNN_SUBJECT_BOOTSTRAP_RESAMPLES=0` (the module's own documented off-switch, gated
`if ... or int(n_bootstrap) <= 0`) rather than skipped by re-implementing the stats.

Usage (CPU-only; submit via submit_reanalysis.sbatch, not on the login node):
    python reanalyze_from_cache.py --alias ctt --dir <the run_rl_rollout.py --out dir for ctt>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Same alias -> subfolder-name mapping as run_rl_rollout.py's BASELINES.
ALIAS_SUBDIR = {
    "ctt": "ForagingCompareThreshold",
    "bari": "QLearning_L1F1_CK1_softmax",
    "hattori": "QLearning_L2F1_softmax",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True, choices=sorted(ALIAS_SUBDIR))
    ap.add_argument("--dir", required=True, help="the --out/<alias> dir from run_rl_rollout.py")
    ap.add_argument("--wrapper", default="/home/han.hou/code/aind-disrnn-wrapper/code")
    args = ap.parse_args()

    sys.path.insert(0, args.wrapper)
    from post_training_analysis.generative_analysis import (
        run_post_training_analysis_from_histories,
        ResolvedModelRun,
    )
    import pickle

    d = Path(args.dir).expanduser().resolve()
    subdir = ALIAS_SUBDIR[args.alias]
    sim_path = d / "analysis" / subdir / "simulated_session_history.pkl"
    animal_path = d / "analysis" / "animal_session_history.pkl"
    resolved_run_path = d / "resolved_run.json"

    print(f"=== reanalyzing {args.alias} from cache ({d}) ===", flush=True)
    for p in (sim_path, animal_path, resolved_run_path):
        if not p.exists():
            raise FileNotFoundError(f"missing cached input: {p}")

    with sim_path.open("rb") as f:
        simulated_sessions = pickle.load(f)
    with animal_path.open("rb") as f:
        animal_sessions = pickle.load(f)
    resolved_run = ResolvedModelRun(**json.loads(resolved_run_path.read_text()))

    print(f"  {len(animal_sessions)} animal / {len(simulated_sessions)} simulated sessions "
          "loaded from cache -- NOT re-simulating", flush=True)

    out_dir = d / "analysis" / subdir  # overwrite in place; same layout run_rl_rollout.py used
    result = run_post_training_analysis_from_histories(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
        output_dir=out_dir,
        resolved_run=resolved_run,
        session_partitions=["combined"],
    )
    (d / "reanalysis_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
