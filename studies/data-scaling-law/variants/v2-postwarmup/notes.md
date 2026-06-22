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

**W&B Sweep (optional, if you want the sweep UI):** confirmed 2026-06-22 that
offline-synced runs **cannot** be attached to a sweep retroactively (`wandb sync` has no
`--sweep` flag, ignores `WANDB_SWEEP_ID`, and `run.sweep` isn't API-settable). A real
sweep requires runs to init **online** with `WANDB_SWEEP_ID` set at creation. So for a
v2 sweep: keep it online (it's configured online here) and inject `WANDB_SWEEP_ID` per
task in the launcher (a small `launch_beaker_resumable.py` addition) — but note an
offline *fallback* would drop sweep membership for that run. Otherwise rely on the
**group** for comparison (works regardless of online/offline), as v1 does.

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
