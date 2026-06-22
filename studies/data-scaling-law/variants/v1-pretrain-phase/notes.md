# Variant v1 — pretrain-phase (first run, 2026-06-22)

First data-scaling run. **Completed; 15/15 in W&B.**

- **Config:** GRU H128, `session_encoding_type=scalar` (session conditioning ON),
  `n_steps=300000`, `lr=1e-5`, `length_bucketing=true`,
  `early_stopping{enabled, min_delta 0.003, patience 2, overfit_guard 0.01}`.
- **Grid:** `data.subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}` × `seed ∈ {0,1,2}`
  → D ≈ {10, 30, 100, 300, 614}.
- **Code:** `WRAPPER_REF=bdb326d`, `DISPATCHER_REF=study/data-scaling-law`.
- **Compute:** onprem-H200, `WANDB_MODE=offline` (a transient cluster→W&B outage made
  online init time out; offline sidestepped it — see root README status log). Full 8+4
  quota: 11 preemptible + 4 allocated (largest-D).
- **W&B:** project [`AIND-disRNN/mice_data_scaling`](https://wandb.ai/AIND-disRNN/mice_data_scaling),
  group `mice-data-scaling-gru`, runs `mice-data-scaling-gru-*-r1` (synced offline → W&B
  under `-r1` ids since the originals had been deleted).
- **Beaker experiment:** `01KVQ7EJ3C5YJ8FJVNJB8C8N36`.

## Result — held-out generalization saturates

| D | mean held-out LL | per-seed |
|---|---|---|
| ~10  | 0.7219 | 0.7202, 0.7216, 0.7238 |
| ~30  | 0.7250 | 0.7248, 0.7250, 0.7252 |
| ~100 | 0.7262 | 0.7260, 0.7264, 0.7264 |
| ~300 | 0.7267 | 0.7265, 0.7267, 0.7268 |
| ~614 | 0.7268 | 0.7266, 0.7268, 0.7268 |

Rises ~0.722→0.727 from D≈10→100, then flat to 614 (~+0.005 over 60× more mice).

## ⚠️ Caveat (why v2 exists)

All runs **early-stopped at step ~40k**, inside *pretrain* (session-conditioning warm-up
is 90k→150k), so session conditioning **never engaged**. This is the "more mice, no
session conditioning" regime — saturation is least surprising here. v2-postwarmup re-runs
to let session conditioning actually engage.
