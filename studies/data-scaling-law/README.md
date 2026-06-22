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
- y-axis: held-out-mouse likelihood, computed **OFFLINE** (post-hoc) from each run's
  saved checkpoint — fine-tune only the subject embedding on a held-out mouse's
  sessions, then predict its other sessions. Training itself is kept lean:
  `checkpoint_run_heldout_eval=false` and `auto_heldout_finetune.enabled=false`
  (no per-checkpoint or end-of-training held-out passes).

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

## Offline held-out evaluation (the y-axis)

After the 15 training runs finish, for each run mount its `/results` dataset and run
the held-out fine-tune+test from its best checkpoint (same function the in-training
`auto_heldout_finetune` used, just driven standalone):

```bash
python code/run_heldout_subject_finetuning.py --config configs/config_heldout_subject_finetuning.yaml \
   source_run.run_dir=<mounted /results/run> source_run.checkpoint_policy=best_eval
# (or: python run_analysis.py finetune --config ...)
```

This is a separate Beaker job per run (reuses the resume remount pattern). It fine-tunes
only the held-out subjects' embeddings and reports their likelihood — re-runnable if we
change the protocol, without retraining.

## Analyze

```bash
.venv/bin/python studies/data-scaling-law/analyze_scaling.py
```
Writes `scaling_results.csv` (+ `scaling_curve.png` if matplotlib present): held-out
likelihood vs actual #training mice (`len(resolved_subject_ids)`), mean over seeds,
power-law fit `L = E + (Dc/D)^α`. Sources the held-out number from the offline
fine-tune outputs (JSON / its W&B run), keyed back to each (D, seed) training run.

## Status log

- 2026-06-22: study scaffolded; nested-sampling patch landed in wrapper
  (`study/data-scaling-law`, `41efc09`); resume path re-validated on Beaker.
