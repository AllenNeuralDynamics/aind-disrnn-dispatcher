#!/usr/bin/env python
"""Roll the three per-mouse RL baselines out generatively, for r4's reference lines.

r1 compares the disRNN's held-out LIKELIHOOD against three classical RL baselines and finds the
disRNN *loses* to compare-to-threshold at D=614. r4 then shows the disRNN's generative BEHAVIOR
trails the GRU's. The obvious next question — *does the disRNN behave more mouse-like than a simple
RL model, or does it lose there too?* — has never been asked: the RL baseline runs carry only
likelihood metrics (no switch/history scalars), and study 01 never crossed its RL report (r8,
likelihood) with its generative one (r9, GRU-only).

This closes that. For each baseline run it simulates every mouse's sessions from that mouse's own
fitted RL parameters and computes the SAME behavioral curves the disRNN rollouts produced.

WHY THIS IS COMPARABLE. It goes through the wrapper's own `run_baseline_rl_post_training_analysis`,
which calls `build_curriculum_matched_task()` — the identical task construction the disRNN rollouts
used (including the off-curriculum / Random Walk fix from wrapper #60). Same tasks, same trial
counts, same seeded rollouts, same statistics. Nothing here re-implements the analysis.

TWO ADAPTERS ARE NEEDED, both mechanical:

1. `hydrate_model_dir()` refuses a baseline-RL run: it demands `checkpoints/step_*`, which such a
   run has no reason to have (its "model" is a table of fitted parameters). We assemble the
   model_dir directly instead — `resolve_model_run()` itself handles `model_type=baseline_rl` fine.

2. `run_baseline_rl_post_training_analysis` expects the EXTERNAL per-session fitting schema
   (`nwb_name`, `agent_alias`, `params`, ...). Our baseline runs fit each mouse ONCE, and store the
   result per subject (`subject_fit_metrics.pkl`: one row per mouse, params as flat columns). We
   expand each subject's parameters across that subject's sessions. This is faithful, not a fudge:
   the fit is per-subject by construction, so the parameters ARE constant across a mouse's sessions.

   `log_likelihood` / `LPT` are required by the schema but unused by the rollout (only `params`,
   plus `n_trials` / `curriculum_name` / `ses_idx`, which come from the ANIMAL session join). We
   pass the subject-level values through rather than invent per-session ones, and never report them.

Usage (CPU-only; run under srun, not on the login node):
    python run_rl_rollout.py --alias ctt --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# The three baseline runs behind r1's RL numbers, in AIND-disRNN/mice_data_scaling.
BASELINES = {
    #  study label            W&B run   wrapper alias preset        r1's held-out LL
    "ctt":     dict(run_id="lmg1i9yd", alias="ForagingCompareThreshold",  heldout_ll=0.7170),
    "bari":    dict(run_id="bg3nzqz9", alias="QLearning_L1F1_CK1_softmax", heldout_ll=0.7149),
    "hattori": dict(run_id="unhmbrk4", alias="QLearning_L2F1_softmax",     heldout_ll=0.7127),
}
ENTITY, PROJECT = "AIND-disRNN", "mice_data_scaling"

# Columns in subject_fit_metrics.pkl that are bookkeeping, not fitted parameters. Whatever remains
# IS the parameter vector -- derived by exclusion so a model with different free params still works.
_NON_PARAM_COLUMNS = {
    "agent_class", "agent_kwargs", "subject_id", "subject_index", "curriculum_name",
    "num_train_sessions", "num_eval_sessions", "num_train_trials", "num_eval_trials",
    "train_likelihood", "eval_likelihood", "train_total_log_likelihood", "train_total_trials",
    "eval_total_log_likelihood", "eval_total_trials", "log_likelihood_train", "LPT_train",
    "AIC", "BIC", "n_free_params",
}


def build_model_dir(run_id: str, dest: Path) -> Path:
    """Assemble a model_dir from the run's training-output artifact.

    Not `hydrate_model_dir()`: that helper requires checkpoints/step_*, which a baseline-RL run
    does not have. Everything else (inputs.yaml from the run config, outputs/ from the artifact) is
    identical, and `resolve_model_run()` accepts the result.
    """
    import wandb
    from post_training_analysis.wandb_model_dir import fetch_run_config, _write_inputs_yaml

    model_dir = dest / "run"
    outputs_dir = model_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    model_type = _write_inputs_yaml(fetch_run_config(ENTITY, PROJECT, run_id), model_dir)
    if model_type != "baseline_rl":
        raise ValueError(f"run {run_id} is model.type={model_type!r}, expected baseline_rl")

    # NB: the artifact is named with HYPHENS ("baseline-rl-output-..."), unlike the model_type
    # ("baseline_rl") that names the disRNN/GRU ones ("disrnn-output-...").
    ref = f"{ENTITY}/{PROJECT}/baseline-rl-output-{run_id}:latest"
    print(f"  downloading {ref}", flush=True)
    wandb.Api().artifact(ref, type="training-output").download(root=str(outputs_dir))
    return model_dir


def build_fitting_df(model_dir: Path, animal_sessions: pd.DataFrame, alias: str) -> pd.DataFrame:
    """Expand per-subject fitted params across each subject's sessions (see module docstring)."""
    subject_fits = pd.read_pickle(model_dir / "outputs" / "subject_fit_metrics.pkl")
    param_columns = [c for c in subject_fits.columns if c not in _NON_PARAM_COLUMNS]
    if not param_columns:
        raise ValueError(f"no fitted-parameter columns found in subject_fit_metrics.pkl for {alias}")
    print(f"  {len(subject_fits)} subject fits | params: {param_columns}", flush=True)

    params_by_subject = {
        str(row["subject_id"]): {c: float(row[c]) for c in param_columns}
        for _, row in subject_fits.iterrows()
    }
    ll_by_subject = {
        str(row["subject_id"]): (row.get("log_likelihood_train"), row.get("LPT_train"))
        for _, row in subject_fits.iterrows()
    }

    rows = []
    for _, session in animal_sessions.iterrows():
        subject_id = str(session["subject_id"])
        params = params_by_subject.get(subject_id)
        if params is None:            # a mouse with no fit: drop it rather than guess
            continue
        log_likelihood, lpt = ll_by_subject[subject_id]
        rows.append(
            {
                "nwb_name": session["nwb_name"],
                "agent_alias": alias,
                "params": params,
                "n_trials": int(session["n_trials"]),
                # required by the schema, unused by the rollout -- subject-level, NOT per-session.
                "log_likelihood": log_likelihood,
                "LPT": lpt,
            }
        )
    fitting_df = pd.DataFrame(rows)
    n_dropped = len(animal_sessions) - len(fitting_df)
    if n_dropped:
        print(f"  WARNING: {n_dropped} sessions dropped (subject has no fit)", flush=True)
    print(f"  fitting_df: {len(fitting_df)} session rows "
          f"over {len(params_by_subject)} subjects", flush=True)
    return fitting_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True, choices=sorted(BASELINES))
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--wrapper", default="/home/han.hou/code/aind-disrnn-wrapper/code")
    args = ap.parse_args()

    sys.path.insert(0, args.wrapper)
    from post_training_analysis import run_baseline_rl_post_training_analysis
    from post_training_analysis.generative_analysis import (
        load_animal_session_history,
        resolve_model_run,
    )

    spec = BASELINES[args.alias]
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.alias} ({spec['run_id']}, alias={spec['alias']}) ===", flush=True)

    model_dir = build_model_dir(spec["run_id"], out)
    resolved_run = resolve_model_run(model_dir, split="train", checkpoint_policy="final")
    animal_sessions = load_animal_session_history(resolved_run)
    print(f"  {len(animal_sessions)} animal sessions", flush=True)

    fitting_df = build_fitting_df(model_dir, animal_sessions, spec["alias"])
    fitting_df_path = out / "fitting_df.pkl"
    fitting_df.to_pickle(fitting_df_path)

    resolved_run_path = out / "resolved_run.json"
    resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))

    print("  simulating (this is the same code path the disRNN rollouts used)...", flush=True)
    result = run_baseline_rl_post_training_analysis(
        resolved_run_path,
        fitting_df_path,
        model_aliases=[spec["alias"]],
        output_dir=out / "analysis",
        n_rollouts_per_session=1,
    )
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  done -> {out / 'analysis'}", flush=True)


if __name__ == "__main__":
    main()
