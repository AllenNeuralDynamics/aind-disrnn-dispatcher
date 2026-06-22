# Variant v2 — post-warm-up (DRAFT, not yet launched)

**Goal:** answer the question v1 couldn't — does training on more mice help held-out
generalization *when session conditioning is actually active*? v1 early-stopped at ~40k,
inside pretrain, so session conditioning (warm-up 90k→150k) never engaged.

**Change from v1:** train *through* warm-up. Draft uses `early_stopping.enabled=false`
(option a — trains to `n_steps=300000`; ~20–30h/run at ~0.24–0.36 s/step). Alternative
(option b): add a `start_after_step` knob to the wrapper early-stopping so it only checks
after ≥150k — keeps the compute-saver but needs a wrapper change. **Decide before launch.**

**W&B:** same project `AIND-disRNN/mice_data_scaling`, distinct group
`mice-data-scaling-v2-postwarmup` (so it sits beside v1 for comparison).

**Compute:** onprem-H200, `WRAPPER_REF=6ede321` (retry→offline-fallback safeguard), **online**
by default (the v1 init failures were a transient outage; safeguard auto-falls-back if it
recurs). Same full 8+4 quota split (post-edit the rendered spec to flip the 4 largest-D to
`{priority:normal, preemptible:false}`, as in v1).

**Launch (when finalized):**
```bash
python code/launch_beaker_resumable.py \
  --sweep studies/data-scaling-law/variants/v2-postwarmup/sweep.yaml \
  --experiment studies/data-scaling-law/variants/v2-postwarmup/experiment.yaml \
  --output-dir /tmp/dsl_v2
```
