# Generative behavioral-match vs D — 2026-06-23

2nd-order validation: roll the GRU out as an agent and compare its switch-triggered
behavioral curve (post-switch choice by reward × run-length) to the real mouse. Metric =
subject-mean **correlation** (the corr~0.96 headline) + subject-balanced **RMSE**, on the
COMBINED session partition. 30 runs (5 D × 3 seeds × v1/v2), wrapper 916d3b4.

| D | v1 corr | v2 corr | v1 RMSE | v2 RMSE |
|---|---|---|---|---|
| 10  | 0.9577 | 0.9675 | 0.0383 | 0.0416 |
| 30  | 0.9756 | 0.9717 | 0.0394 | 0.0401 |
| 100 | 0.9774 | 0.9834 | 0.0374 | 0.0366 |
| 300 | 0.9777 | 0.9833 | 0.0364 | 0.0376 |
| 614 | 0.9789 | 0.9802 | 0.0362 | 0.0379 |

**Findings:**
- **Match is high at every D** (corr 0.96–0.98): the model reproduces the 2nd-order
  switch dynamics well even at D=10. This is the corr~0.96 headline, now resolved across D.
- **Modest improvement, saturating by D≈100** — corr rises ~+0.02 from D=10→100 then flattens
  (v1 0.958→0.977→0.979; v2 0.968→0.983→0.980). Same shape as the held-out-LL curve: more
  training mice help a little and plateau early. Not a sustained data-scaling signal.
- **SC (v2) edge is small and mixed:** v2 has slightly higher *correlation* at D≥100
  (0.983 vs 0.977) — SC marginally improves the *shape* match — but v2 RMSE is comparable or
  slightly *higher* than v1, so it doesn't reduce absolute error. Consistent with SC's small
  LL benefit; not a large generative-quality difference.

**Takeaway:** the generative (2nd-order) check corroborates the next-trial-LL story — the model
is already a good behavioral generator at small D, and scaling D buys a small, fast-saturating
gain. It does *not* reveal hidden headroom that LL missed (one of the reasons FUTURE_DIRECTIONS
flags adaptation-efficiency / OOD / lick-level metrics as the headroom-ier axes).

**Caveat (carried):** the rollout task is matched only to the curriculum *family* with default
params, NOT the session's stage-specific params — a confound baked into all D points (see
FUTURE_DIRECTIONS §5 / memory `generative-task-not-stage-matched`). Affects absolute match,
not the vs-D *trend*.

Data: `generative_match.json`; figure `fig_generative_match.png`. W&B groups
`generative-v{1,2}@…` in [mice_data_scaling](https://wandb.ai/AIND-disRNN/mice_data_scaling).
