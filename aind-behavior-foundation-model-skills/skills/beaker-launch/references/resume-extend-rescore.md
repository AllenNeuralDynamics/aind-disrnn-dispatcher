# Resuming, extending & re-scoring runs (three distinct mechanisms)

Don't conflate them. Full lifecycle detail: wrapper
`../aind-disrnn-wrapper/code/TRAINING.md` §1.5 "Run lifecycle & key switches".

## 1. Preemption resume — automatic, WITHIN one experiment

A preempted `preemptible: true` task restarts as the *same* task with the *same*
`/results` dataset, re-finds its latest `checkpoints/step_<N>/train_state.pkl`,
and continues (skipping warmup). Needs `checkpoint_every_n_steps > 0` + the
trainer gate `training.auto_resume` (default; distinct from Beaker's own
`autoResume` spec field). No flags, no new experiment. Each task sets
`DISRNN_RESUMABLE_OUTPUT_DIR=/results/run` so outputs anchor at a fixed path the
restart re-finds. (Beaker-only: HPC `aind` jobs are not preempted.)

## 2. Extend a finished run to a longer horizon — ACROSS experiments

Launch a *new* experiment with
`model.training.restore_from_run_id=<source W&B run name>` (or per-cell env
`DISRNN_RESTORE_FROM_RUN_ID` — env wins, so a sweep can pass a per-cell id) and a
**larger** `n_steps`. Before training, the entrypoint downloads the source run's
`<mtype>-output-<run_id>:latest` artifact (`mtype` ∈ {`disrnn`,`gru`}) into
`outputs/`, so the trainer resumes from its checkpoint and skips warmup.
Trainer-agnostic. **Prereq: the source run must have FINISHED** — its
`training-output` artifact is uploaded once at end of training (not per
checkpoint), so in-progress runs cannot be extended. Fails **loudly** if the
artifact is missing — never silently restarts from scratch. Only seeds when no
local checkpoint exists yet (a preemption restart of the continuation run keeps
its fresher local state).

## 3. Re-score a finished run's held-out stage only — no re-training

`python code/resume_heldout_beaker.py --run-id <wandb_run_id>` (from the wrapper
repo, inside a Beaker container that reaches GCS + W&B). Runs the held-out
fine-tune ONLY off the downloaded checkpoint tree, reads every knob (seed,
`checkpoint_policy`, held-out set, finetune `n_steps`/`lr`) from the SOURCE run's
own config, and re-injects `heldout/*` back into the ORIGINAL W&B run. Use it to
backfill metrics added *after* a run trained (e.g. the 3-way ignore-class
precision/recall/F1/PR-AUC).

This is the **exact-reproduction** path: unlike (2)'s restore — which resumes the
training entrypoint and redraws a fresh held-out set off the restored
checkpoints — this reproduces the source run's original held-out numbers.

The HPC original is `code/resume_heldout.py`
(`--model-dir <dir> --wandb-run-id <id>`, run on a compute node that can read the
checkpoint tree + reach W&B); both live under the wrapper's `code/`.
