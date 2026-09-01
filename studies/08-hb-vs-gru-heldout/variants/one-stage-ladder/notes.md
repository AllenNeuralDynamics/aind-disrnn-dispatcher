# Variant: one-stage-ladder

One-stage three-level `HB-Hattori2019` fits across study 01's D dial, scored on the
held-out cohort with matched conditioning.

| | |
|---|---|
| **W&B group** | `hb-one_stage@20260829-181251` |
| **W&B project** | `mice_data_scaling` — **study 01's project, not a new one** (see deviation below) |
| **Runs** | `hb-one_stage-D{10,30,101}-s0` |
| **SLURM** | superseded — see the launch record; no rung has run on the current code |
| **Status** | not yet run. The entrypoint moved to `run_hpc` after those jobs, so they do not reflect this variant |

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
- **No `sweep.yaml`.** These are direct `sbatch` submissions rather than a W&B sweep, so
  there is no sweep config to record; `production.sbatch` is the launch surface.

## Known limits

Held-out scoring originally ran one adaptation fit per subject, sequentially: roughly 4 h
per rung over the 153-subject cohort, which forced production down to `matched` + `k=0`.

Adaptation and session scoring are both batched now, so a rung costs minutes and
`production.sbatch` runs the full `k ∈ {0,1,2,4,8}` sweep from the model config rather than
overriding it. The `--few-shot-k 0` in the superseded launch record reflects the old cost.
