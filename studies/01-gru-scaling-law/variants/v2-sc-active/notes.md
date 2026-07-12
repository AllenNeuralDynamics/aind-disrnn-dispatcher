# Variant v2 — sc-active (finalized 2026-06-22, not yet launched)

**Goal:** answer what v1 couldn't — does training on more mice help held-out
generalization *when session conditioning is actually active and trained*? v1
early-stopped at ~40k, inside pretrain (λ=0), so SC never engaged.

## Design — lambda forward + gated early stop

Grounded in prior runs (see the study README "Prior evidence" + W&B `meta.note`):
- **Plain GRU (λ=0) converges ~30–40k** (v1: D=10 valid-loss bottoms ~20k then overfits;
  D=614 near-flat by 40k). So v1's `pretrain=90k` (30% of n_steps=300k) wasted ~50k steps.
- **SC engaged adds little / overfits at small D** — Po-chen's D=10 (`mice_multisubject_
  train10`): SC helps small H, **hurts H128** within-population; eval *declines* after λ→1.
  None measured held-out-*mouse* generalization → that's v2's open question.

Therefore:
- `session_n_pretrain_steps=30000`, `session_n_warmup_steps=20000` → **full λ=1 at 50k**
  (forward from v1's 150k).
- `n_steps=150000` — headroom so the gated early stop can operate.
- **Early stopping ON but gated:** `start_after_step=70000` (20k of full-λ training first),
  `metric=eval_likelihood`, `patience=2`, `min_delta=0.003`, `overfit_guard=0.01`. Catches
  the small-D post-SC overfit while letting large-D keep training toward 150k. Needs the
  wrapper knob `early_stopping.start_after_step` (WRAPPER_REF below).

## Compute / tracking

- **WRAPPER_REF=`65c3350`** (branch `study/data-scaling-law`): cd72d1e (retry→offline
  safeguard + provenance) **+ meta.note + early_stopping.start_after_step**.
- **DISPATCHER_REF=`study/data-scaling-law`**. Cluster onprem-H200, **online** (safeguard
  auto-falls-back to offline on a flaky init; sync per the root README if so).
- Full 8+4 quota: post-edit the rendered spec to flip the 4 largest-D to
  `{priority:normal, preemptible:false}` (as in v1).
- W&B project `AIND-disRNN/mice_data_scaling`, group `v2-sc-active@<launch_id>` (launcher),
  beside v1 for comparison. **Always pass `--note`** so the run records its intent.

**W&B sweep not usable for this pipeline** (tested 2026-06-22): a sweep is agent-owned —
externally-created resumable runs can't attach (sync ignores `WANDB_SWEEP_ID`; injecting it
into a PENDING `wandb.sweep()` 404s), and the `wandb agent` route doesn't support our
per-grid-point checkpoint-resume. Use the **group** for comparison (Runs table / parallel
coords give the same cross-run view).

## ETA (from prior runs)

Step-rate: v1 λ=0 steps ~0.26–0.42 s/step (rises with D); SC-active steps cost more
(`mice_scaling` D=614 ~0.55–0.61 s/step). v2 has per-checkpoint heldout-eval OFF →
blend ~0.45–0.55 s/step. Most runs early-stop ~90–120k after the 70k gate.
- small-D ~10–13 h · mid-D ~13–16 h · large-D ~16–18 h (worst case full 150k ≈ ~22 h).
- **All 15, near-concurrent on the 8+4 quota → ~16–18 h wall-clock** (worst case ~24–30 h
  for ~2 waves / a 150k-capped large-D run). Preemption auto-resumes in place.

## Launch (when ready)

```bash
python code/launch_beaker_resumable.py \
  --sweep studies/data-scaling-law/variants/v2-sc-active/sweep.yaml \
  --experiment studies/data-scaling-law/variants/v2-sc-active/experiment.yaml \
  --note "v2-sc-active: does more training data help held-out-mouse generalization with session conditioning ACTIVE (lambda fwd: full SC @50k) + gated early-stop @70k? v1 saturated ~0.727 but SC never engaged." \
  --output-dir /tmp/dsl_v2
```
