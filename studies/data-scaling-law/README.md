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

### Syncing the offline W&B runs (this study runs `WANDB_MODE=offline`)

The runs log offline to `/results/wandb/offline-run-*`, and `/results` is **each Beaker
task's result dataset — persisted to Beaker/S3, not a local path**. So to get the runs
into W&B (for `analyze_scaling.py`, which reads the W&B API) you must first **fetch the
result datasets from Beaker**, then `wandb sync`:
```bash
# pull all task result folders from S3 (one folder per task)
beaker experiment results 01KVQ7EJ3C5YJ8FJVNJB8C8N36 -o /tmp/dsl_results
# sync each task's offline run into W&B (login node reaches W&B fine)
for d in /tmp/dsl_results/*/wandb/offline-run-*; do wandb sync "$d"; done
```
(Per-task result-dataset id is also at `jobs[].result.beaker` in
`beaker experiment get <exp> --format json`; fetch one with `beaker dataset fetch <id>`.)
Authoritative metrics also live in each dataset's `/results/run/outputs`
(`output_summary.json`, checkpoint metrics) independent of W&B.

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
  (`1265b07`) and relaunched as experiment `01KVQ3EXZ6PNVQETACT42TGB58` (15
  autoResume tasks, onprem-H200).
- 2026-06-22: concise W&B run names — replaced the full subject-ID list in the run
  name with the subject count (`_n<count>subj`; full list stays in
  `config.resolved_subject_ids`), wrapper `ed8f50e`. Relaunched as
  `01KVQ42H3ZT0EJQGXD3KRBKV3V` — but that launch surfaced two W&B failure modes:
  (1) reusing a deterministic `WANDB_RUN_ID` from a cancelled run + a changed
  `wrapper_commit` → `ConfigError`; (2) 13 simultaneous `wandb.init` calls
  overwhelming the backend → 90s init timeout (only 1/15 runs came up).
- 2026-06-22: hardened `start_wandb_run` (wrapper `ef862fd`): per-run staggered
  init, `init_timeout=300`, retry+backoff, and `allow_val_change=True` on the
  SHA stamp (knobs `WANDB_INIT_TIMEOUT`/`WANDB_INIT_STAGGER`). Relaunched as
  `01KVQ4VSN5HXB91YH0MF7EZ5K0` — `allow_val_change` killed the ConfigError, but
  all 13 still hit `wandb.init` within a 20s window, hung the full 300s, and
  retried in sync (0/15 runs came up). W&B itself was healthy (status page green,
  direct API reads <1s) — the bottleneck is the onprem cluster's shared egress
  under a 15-way simultaneous init burst (single/few runs connect fine).
- 2026-06-22: the 180s stagger relaunch (`01KVQ5BH567MF1FEG8YS7Y5M9N`) **refuted
  the concurrency hypothesis** — inits spread over 146s, yet even near-solo inits
  (40s isolation) still timed out at 300s. Real root cause: the **onprem-H200
  cluster's network path to W&B run-creation degraded ~07:49** (W&B status green,
  my-machine API reads <1s, bench connected fine at 07:08). Not W&B, not
  concurrency, not code (auth succeeds; the run-*create* call hangs).
- 2026-06-22: reverted the speculative stagger/timeout/retry (wrapper `bdb326d`;
  kept `allow_val_change` — a real fix). **Moved the sweep to AWS L40s**
  (`octo-hub-aws-l40s`): H128 fits 48GB, reaches the AWS DB, and AWS→W&B works.
  Node = 1 machine / 4 GPU slots / ~373 GiB → `memory=90GiB`, `cpuCount=8` packs
  4 concurrent (rest queue + drain via autoResume). **Relaunched as experiment
  `01KVQ682F0XPAH4QD58SAKQ4R7`** (`WRAPPER_REF=bdb326d`). 3-h cron repointed.
  Throughput note: only ~4 run at once on the single L40s node, so 15 drain in
  waves (early stopping shortens each).
- 2026-06-22: ran offline on L40s (`01KVQ6K7…`), trained fine. Then probes settled
  the real root cause: on **onprem-H200 a single `wandb.init` succeeds (~1.5s) but
  14 concurrent ones all time out** — a W&B **run-creation throttle under a
  concurrent-init burst**, not a dead path. (The earlier "L40s also fails even
  staggered" was a red herring: L40s has a *separate* W&B-reachability problem, so
  staggering couldn't be validated there — running that test on L40s instead of
  H200 wrongly sank the concurrency theory.) Online needs init concurrency limited
  (stagger/serialize); offline sidesteps it entirely since init is local.
- 2026-06-22: final config — **onprem-H200 + `WANDB_MODE=offline` + full 8+4 quota**.
  The resumable launcher only uses the 8 preemptible slots (autoResume), so the
  rendered spec is post-edited to flip the **4 largest-D runs** (ratio 1.0 ×3 seeds
  + ratio 0.489 seed 0) to **allocated/non-preemptible** (uses the 4 allocated
  slots; no autoResume, but protected — relaunch-to-resume if one dies).
  **Experiment `01KVQ7EJ3C5YJ8FJVNJB8C8N36`** (15 tasks, ~12–14 concurrent). Offline
  runs are persisted to each task's **Beaker result dataset (S3)**, not a local path
  — `beaker experiment results <exp>` to fetch, then `wandb sync` (see "Syncing the
  offline W&B runs"). Track health via Beaker logs, not the live W&B project.
- 2026-06-22: wrapper `6ede321` — `start_wandb_run` now tries online with jittered
  retry, then **falls back to `WANDB_MODE=offline`** so training never blocks on a
  flaky online run-creation handshake (knobs `WANDB_INIT_ATTEMPTS`/`WANDB_INIT_TIMEOUT`;
  post-finish sync deliberately not added — it'd run on the cluster where the same
  path may be down). NOT adopted by the running sweep: it stays pinned to
  `WRAPPER_REF=bdb326d` + explicit `WANDB_MODE=offline` so all 15 tasks (and any
  caretaker relaunch of a dead allocated task) behave identically. **Future launches**
  can bump `WRAPPER_REF=6ede321` and drop `WANDB_MODE=offline` from the template to get
  adaptive online-with-offline-fallback automatically.
