# Variant nxd-grid — N×D joint-scaling grid (launched 2026-06-23)

**Goal:** test the N (model capacity = `hidden_size`) × D (#mice) interaction for
held-out-mouse generalization. We ALREADY have **H128 × all D** from variant
`v2-sc-active`, so this variant runs only the OTHER hidden sizes; the H128 column is
folded back in at analysis time for the full joint N×D view.

## Grid (27 tasks)

`hidden_size ∈ {16, 64, 256}` × `subject_ratio ∈ {0.016, 0.163, 1.0}` (D≈10/100/614)
× `seed ∈ {0, 1, 2}`.

ALL other knobs identical to `v2-sc-active`: scalar SC, lambda forward
(`session_n_pretrain_steps=30000`, `session_n_warmup_steps=20000` → full λ=1 @50k),
`n_steps=150000`, `lr=1e-5`, gated early stop (`enabled=true`, `start_after_step=70000`),
`length_bucketing=true`, per-checkpoint heldout eval OFF
(`checkpoint_run_heldout_eval=false`).

## Compute / tracking

- **WRAPPER_REF=`4f296807f4ea06f9a58afa4eeb0553c220db4726`** (study/data-scaling-law:
  per_subject_likelihood.json + early_stopping.start_after_step).
- **DISPATCHER_REF=`study/data-scaling-law`**. Cluster onprem-H200, gpuCount 1 / cpu 16
  / mem 256GiB, priority low / preemptible (autoResume on eviction). Online by default.
- W&B project `AIND-disRNN/mice_data_scaling`, group **`nxd-grid@20260623-092311`**,
  beside `v2-sc-active` (the H128 column) for the joint N×D comparison.
- **Beaker experiment:** `01KVTMNY1XB2GRAVGYD7FJ2GV6`
  (https://beaker.org/ex/01KVTMNY1XB2GRAVGYD7FJ2GV6).

## Status

Launched 2026-06-23 ~09:23 PT. At submit: 27 jobs, 14 scheduled / 13 queued. *Note*: the first launch at `092311` was cancelled at 16:34 UTC after the cheap zero/few-shot finished freeing the queue; *relaunched* as `nxd-grid@20260623-102649` (exp `01KVTRAF21E73WDWEX8MDRYM9E`).

**Completed 2026-06-24 ~12:30 UTC**: 27/27 jobs OK (1 preemption auto-resumed once on task -000). All written to W&B group `nxd-grid@20260623-102649`. Analysis in `studies/data-scaling-law/analysis/nxd_scaling.{py,json}` + `fig_nxd_scaling.png` + `nxd_scaling_verdict.md`. Headline result folded into `analysis/reports/r7-nxd-joint-scaling-grid.md`. TL;DR: D saturates at every N, weak Chinchilla-style N×D interaction (gain from N=16→256 grows from +0.004 at D=10 to +0.006 at D=614, ×1.7), single irreducible floor E≈0.729.

## D≈30 gap-fill launch (g6e L40S)

Launched 2026-06-24 14:11 PT from Allen HPC using the `disrnn-cpu` conda env.

- **Beaker experiment:** `01KVXQHTSH0780FFBCJAGAXA2E`
  (https://beaker.org/ex/01KVXQHTSH0780FFBCJAGAXA2E).
- **W&B:** project `AIND-disRNN/mice_data_scaling`
  (https://wandb.ai/AIND-disRNN/mice_data_scaling), group
  **`nxd-grid@20260624-141106`**.
- **Grid:** 9 tasks, `hidden_size ∈ {16, 64, 256}` × `subject_ratio=0.049`
  (D≈30) × `seed ∈ {0, 1, 2}`. H128 at D≈30 is still reused from
  `v2-sc-active`.
- **Resources:** `ai1/octo.ai-aws-g6e`, low / preemptible / autoResume,
  gpuCount 1 / cpu 12 / mem 90GiB.
- **Launch verification 2026-06-24 14:13 PT:** all 9 jobs were scheduled and
  started; Beaker assigned one GPU per task with 90GiB / 12 CPU. The three H256
  D≈30 jobs reached "Before Training" without the earlier `RESOURCE_EXHAUSTED`
  / OOM signature. Observed H256 full dataset shapes were `(1238, 814, 4)`,
  `(1043, 970, 4)`, and `(1505, 749, 4)`.

Completion and the `nxd_scaling.py` D=30 analysis fold-in are still pending.
