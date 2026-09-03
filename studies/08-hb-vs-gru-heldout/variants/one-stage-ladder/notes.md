# Variant: one-stage-ladder

One-stage three-level `HB-Hattori2019` fits across study 01's D dial, scored on the
held-out cohort with matched conditioning.

| | |
|---|---|
| **W&B project** | `mice_data_scaling` — **study 01's project, not a new one** (see deviation below) |
| **Launch surface** | Beaker (`sweep_beaker.yaml` + `experiment_beaker.yaml`); `production.sbatch` still carries pre-rename paths |
| **Status** | D10 complete on the current code. D30 in flight. D101 and above not yet attempted on the current code |

### Rung state, 2026-09-03

| rung | ratio | state |
|---|---|---|
| D≈10 | 0.016 | complete — path smoke plus three fixed-seed replicates, all exit 0 |
| D≈30 | 0.049 | in flight — Beaker experiment `01M1K112YCMY31GNDKHESW9G6Z`, production sampler 500/500/4 |
| D≈101 | 0.163 | not attempted on the current code |

**The D30 and D101 rungs have never completed.** Both crashed on 2026-08-29 in group
`hb-one_stage@20260829-181251` (SLURM 25489921/22/23), seconds after the trainer started.
That group is the *relaunch* recorded in `launch_record/hpc_ladder_relaunch.json`, whose
record documents only the earlier `ModuleNotFoundError: wandb` failure and the resubmission
— not that the resubmission also died. W&B's retained log lines stop at the JAX
backend-init message and the SLURM `.err` files are on HPC, so the cause was not
recoverable when this was written. Do not read the earlier `hb-one_stage-D{30,101}-s0` runs
as results; they carry no likelihood, no sampler config and no commit stamps.

The three `hb-one_stage-D10-s0` runs that *did* complete on 2026-08-29 predate the
`run_hpc` refactor and the seed fix (dispatcher #108), and their sampler settings were never
recorded in W&B, so they are not comparable with anything run since.

## What differs

Only `data.subject_ratio`: 0.016 / 0.049 / 0.163, giving D ≈ 10 / 30 / 101 against the
~614 pool. Everything else matches study 01's `mice_snapshot_scaling` config — same
snapshot `20260603`, curricula, `min_sessions=10`, `heldout_every_n=5`,
`mature_only=false`, `ignore_policy=exclude`, `eval_every_n=2`.

## What is scored

`heldout/eval_likelihood` and `heldout/test_likelihood` carry the **matched** rung:
adapt each held-out subject on its `eval_every_n=2` train sessions, score its eval
sessions. That is the same protocol as the GRU's `auto_heldout_finetune` y-axis and the
per-mouse MLE baseline, so the three are directly comparable.

`heldout/few_shot_k0_likelihood` carries zero-shot as a second, different claim: what the
cohort prior alone is worth with no data from that subject.

Comparators already on record:

| arm | held-out likelihood |
|---|---|
| per-mouse MLE Hattori (no pooling) | 0.71267 |
| per-mouse MLE CTT (best of family) | 0.71704 |
| GRU D≈30 (seeds 0/1/2) | 0.7248 / 0.7252 / 0.7250 |
| GRU D≈100 (seeds 0/1/2) | 0.7264 / 0.7264 / 0.7260 |

## Deviations from study-conventions

- **Shared W&B project.** The skill says one project per study. This variant logs into
  study 01's `mice_data_scaling` instead, deliberately: the entire point is to land on the
  same scaling curve, and `analyze_scaling.py` reads that project. A separate project would
  make the comparison a manual join.
- **`launch_id` is UTC, not Seattle.** The launcher used `TZ=UTC`; conventions and
  AGENTS.md §7 both call for Seattle. Fixed in the launcher for subsequent launches; this
  group's id is left as-is because the runs are already stamped with it.
- **Resolved, no longer a deviation — sweep configs.** This entry previously read "no
  `sweep.yaml`: these are direct `sbatch` submissions", which stopped being true when the
  Beaker surface landed. `sweep_beaker.yaml` and `sweep_beaker_smoke.yaml` are the sweep
  configs and `launch_beaker_resumable.py` consumes them. Kept as a resolved entry rather
  than deleted, so a reader of the 2026-08-29 records can still see why they have no sweep
  file.

## Known limits

Held-out scoring originally ran one adaptation fit per subject, sequentially: roughly 4 h
per rung over the 153-subject cohort, which forced production down to `matched` + `k=0`.

Adaptation and session scoring are both batched now, so a rung costs minutes and
`production.sbatch` runs the full `k ∈ {0,1,2,4,8}` sweep from the model config rather than
overriding it. The `--few-shot-k 0` in the superseded launch record reflects the old cost.
