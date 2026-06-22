# Study: Mouse data-scaling law

**Question.** Does training a behavior foundation model on more mice improve
prediction of mice it has *never seen*? Within-subject per-trial likelihood
saturates (~0.73–0.75 flat across H128/H256, even H2–H4), so model size is not the
bottleneck and within-subject likelihood is the wrong figure of merit. The
foundation-model metric is **held-out-mouse generalization vs the number of
training mice (D)**.

**Design (minimal; Kaplan/Chinchilla practice — fix N & HPs, vary one axis).**
- Fixed: GRU `hidden_size=128`, `session_encoding_type=scalar` (session conditioning ON), `n_steps=300000`
  (train to convergence — the 100k run was undertrained), `lr=1e-5`, `batch_size=2048`.
- Swept (science axis): `data.subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}`
  → D ≈ {10, 30, 100, 300, ~614} training mice (scalar ratio ⇒ natural curriculum
  composition at every D).
- Replicates: `seed ∈ {0, 1, 2}` → 3 **nested** ladders (loader uses permutation-prefix
  sampling: D=10 ⊂ D=30 ⊂ … for a fixed seed).
- Held-out cohort is **fixed** (`heldout_every_n=5`, every 5th ranked subject per
  curriculum) and constant across all D and seeds — mice never in any training set.
- y-axis: held-out-mouse likelihood from a **single final** held-out fine-tune+test
  at the end of each run (`auto_heldout_finetune`, enabled) — fine-tune only the
  subject embedding on a held-out mouse's sessions, then predict its other sessions.
  Per-checkpoint held-out passes are OFF (`checkpoint_run_heldout_eval=false`) to keep
  eval-during-training minimal. The same fine-tune+test can be re-run offline from the
  saved checkpoint if we change the protocol.

15 runs total (5 D × 3 seeds).

## Launch (resumable, one autoResume task per grid point)

```bash
cd <dispatcher repo>
export PATH="$PWD/.venv/bin:$PATH"   # launcher calls bare `wandb`/`beaker`
python code/launch_beaker_resumable.py \
  --sweep studies/data-scaling-law/sweep_data_scaling.yaml \
  --experiment studies/data-scaling-law/experiment_data_scaling.yaml
```

Each grid point becomes a preemptible Beaker task with `autoResume` (Beaker restarts
it in place with the same `/results`; the trainer resumes from its last full-state
checkpoint). H128 fits both `octo-hub-onprem-h200` (template default; free slots,
reaches the AWS DB) and a 48 GB `octo-hub-aws-l40s`. 15 tasks ride the preemptible
(unallocated) quota and drain in waves; preemption auto-recovers, so even the long
large-D runs need no allocated slots.

Code versions: `WRAPPER_REF=41efc09…` (study/data-scaling-law — nested sampling),
`DISPATCHER_REF=study/data-scaling-law`.

## Pre-launch check

Confirm the largest run converges (the original bug was undertraining): watch
`checkpoint/train_likelihood` of the D≈614 task — it should plateau ≳0.75 by 300k. If
still climbing, raise `lr`/`n_steps`. Also confirm the smallest D (ratio 0.016) still
spans all three curricula in the loader's `[select 6/6]` log (rounding can drop a
small curriculum; raise the smallest ratio if so).

## Analyze

```bash
.venv/bin/python studies/data-scaling-law/analyze_scaling.py
```
Writes `scaling_results.csv` (+ `scaling_curve.png` if matplotlib present): held-out
likelihood (`heldout/eval_likelihood` / `heldout/test_likelihood`, from each run's final
`auto_heldout_finetune`) vs actual #training mice (`len(resolved_subject_ids)`), mean over
seeds, power-law fit `L = E + (Dc/D)^α`.

### Optional: re-run held-out offline

The same fine-tune+test can be re-run from a saved checkpoint without retraining (e.g. to
change the protocol) — mount the run's `/results` and:
```bash
python code/run_heldout_subject_finetuning.py --config configs/config_heldout_subject_finetuning.yaml \
   source_run.run_dir=<mounted /results/run> source_run.checkpoint_policy=best_eval
```

## Early stopping (manual, consistent across D)

Constant lr=1e-5, **no LR scheduler** (training is stable; scheduler = marginal gain + extra
HP). Stop each run when its within-subject **eval_LL** plateaus, then run held-out offline on
its `best_eval` checkpoint. Criterion (applied identically to every D so the comparison is fair):
- **min-delta ε = 0.003** eval_LL to count as a new best (below this is eval noise),
- **patience = 2 checkpoints (20k steps)** with no new best → `beaker job cancel` that run,
- hard overfit guard: cancel if eval_LL drops > 0.01 below the running best.
`best_eval` is the safety net — the held-out fine-tune always uses the best checkpoint, so ε
only controls compute saved, not result quality. (Within-subject eval saturates ~0.74 across
H2–H256 — a noise/feature ceiling, not capacity; see TODO.)

## TODO / follow-ups

- **Hidden-size scan on the HELD-OUT metric.** Within-subject eval is flat H2→H256 (capacity-
  saturated), so model size looks irrelevant there — but capacity may matter more for
  representing a *new* mouse. After the D-curve is in, scan hidden_size {16, 64, 128, 256} at a
  fixed large D and compare **held-out** generalization (not within-subject LL). Keep H128 as
  the default for the main D-sweep.
- N (model-size) × IsoFLOP scaling only if the D-curve shows real headroom.
- **DONE for GRU (PR #41, opt-in / default-off):** native **early stopping**
  (`training.early_stopping {enabled,metric,min_delta,patience,overfit_guard}` — stops at the
  eval-LL plateau and `break`s so finalization + held-out still run, `best_eval` used) and
  **length-bucketed batching** (`training.length_bucketing` + `length_bucket_grid` — draws each
  batch from one grid-rounded session-length bucket and trims the unroll). Padding context:
  sessions median 521 vs T_max 2207 (p95 846), so ~50–75% of unroll compute was padding ⇒ ~2.6x
  predicted. **Measured** (two identical D≈614/H128 runs, W&B `bench_padding`, only `length_bucketing`
  differing): steady-state **0.72 → 0.33 s/step ≈ 2.2x speedup** with an **identical loss curve at
  matched steps** (±0.0015 noise) — slightly under prediction due to grid=128 rounding + fixed
  per-step overhead. Both enablers are on for the 15-run sweep.
- **TODO — port both to disRNN** (needed once disRNN runs are wanted). Early stopping: add the
  same checkpoint-loop hook to `disrnn_trainer`. Length bucketing: the batch sampler
  `session_regularized_training._sample_batch` is **shared**, so it likely "just works" once
  `disrnn_trainer` sets `dataset_train.length_bucketing` on its train set — verify against
  disRNN's warmup/pretrain + session-reg schedule. Defer for now.

## Status log

- 2026-06-22: study scaffolded; nested-sampling patch landed in wrapper
  (`study/data-scaling-law`, `41efc09`); resume path re-validated on Beaker.
- 2026-06-22: fixed resumable-mode `inputs.yaml` bug (`b0c7f11`) that made the
  held-out fine-tune silently skip; re-confirmed held-out eval logs (≈0.70 on a
  tiny D≈30 probe). Launched the 15-run sweep (scalar, lr=1e-5, 300k, lean) as
  experiment `01KVPPMQ38NNT00870Q1QAT0XF` on onprem-H200 (preemptible/autoResume).
  Watching large-D `train_likelihood` to validate lr=1e-5 convergence.
- 2026-06-22: cancelled `01KVPPMQ38…` to fold in PR #41 (early stopping +
  length-bucketing). Padding bench (`bench_padding` W&B project, D≈614, identical
  except `length_bucketing`) confirmed **~2.2x speedup** (steady-state 0.72 →
  0.33 s/step) with an **identical loss curve at matched steps** (±0.0015, noise).
  Enabled `length_bucketing=true` + `early_stopping.enabled=true` in the sweep
  (`1265b07`) and **relaunched as experiment `01KVQ3EXZ6PNVQETACT42TGB58`** (15
  autoResume tasks, onprem-H200). 3-h status cron repointed to the new experiment.
