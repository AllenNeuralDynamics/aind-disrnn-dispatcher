# Resuming, extending & re-scoring runs (three distinct mechanisms)

Don't conflate them. Full lifecycle detail: wrapper
`../aind-disrnn-wrapper/code/TRAINING.md` §1.5 "Run lifecycle & key switches".

## 1. Preemption resume — automatic, WITHIN one experiment

A preempted `preemptible: true` task restarts as the *same* task with the *same*
`/results` dataset, re-finds its latest `checkpoints/step_<N>/train_state.pkl`,
and continues (skipping warmup). Needs `checkpoint_every_n_steps > 0` + the
trainer gate `training.auto_resume` (default; distinct from Beaker's own
`autoResume` spec field). No flags, no new experiment. Each task sets
`BFM_RESUMABLE_OUTPUT_DIR=/results/run` so outputs anchor at a fixed path the
restart re-finds. (Beaker-only: HPC `aind` jobs are not preempted.)

## 2. Extend a finished run to a longer horizon — ACROSS experiments

Launch a *new* experiment with
`model.training.restore_from_run_id=<source W&B run name>` (or per-cell env
`BFM_RESTORE_FROM_RUN_ID` — env wins, so a sweep can pass a per-cell id) and a
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

## 4. Backfill a LOST metric from its surviving table artifact — no GPU at all

**Try this before (3).** If the held-out stage actually *ran* and only the final
summary write was lost (see "exit 0 with a missing metric" in
`scheduling-lessons.md`), the scalar is recoverable **exactly** from the run's own
committed `run_table` artifact — no container, no GPU, no re-training. Re-scoring
via (3) would re-derive a number that already exists.

Check first: does the crashed run have BOTH a `<mtype>-output-<run_id>`
(`training-output`) and a `*-heldoutper_subject_likelihood` (`run_table`) artifact,
each `COMMITTED`? If yes, the held-out stage completed and its full per-subject
output survived.

**Verify the aggregation against natively-logged runs — never assume it.**
`heldout/eval_likelihood` is the **trial-weighted GEOMETRIC** mean of the
per-subject likelihoods (correct: a likelihood is `exp(mean log-lik per trial)`):

```
heldout/eval_likelihood = exp( sum_i n_trials_i * ln(lik_i) / sum_i n_trials_i )
```

Verified on study 06 to <= 5.3e-08 against 5 runs holding both the table and a
native scalar. **The two plausible-looking alternatives are badly wrong**: a simple
mean over subjects is off by ~0.005 and the *arithmetic* trial-weighted mean by
~0.004 — the same magnitude as the effects these studies measure, so a guessed
formula silently poisons the result rather than failing loudly. A backfill script
should re-validate the formula at runtime against native runs and refuse to write
if it drifts.

**Tag what you backfill.** Write a marker alongside the value
(`heldout/eval_likelihood_backfilled=True` + a `_backfill_src` note) so a recovered
number is never mistaken for a natively-logged one, and have the analysis layer
admit *only* flagged rows past its `state == "finished"` filter — any *other*
crashed run's `heldout_ll` is a mid-training incremental value, not a final one.
Worked example: `studies/06-disrnn-operating-point-at-scale/analysis/backfill_lost_heldout.py`
(idempotent, `--dry-run`), which recovered 5 lost D=300 cells.
