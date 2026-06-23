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

Launched 2026-06-23 ~09:23 PT. At submit: 27 jobs, 14 scheduled / 13 queued.
