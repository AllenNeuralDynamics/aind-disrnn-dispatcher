# Variant gru-stage2 — within-subject drift + session conditioning (PLACEHOLDER)

**Status:** scaffolded, sweep NOT yet built. Constructed after Stage 1 recovery is
confirmed, so the Stage-2 grid can reuse whichever (hidden_size, embedding_size)
gave clean between-subject recovery.

**Goal.** Add across-session **drift** to the generator and turn on the GRU's
**session conditioning**, then test whether the model recovers the within-subject
trajectory (not just the static subject centroid).

## Planned generator change (Stage 2)

Populate `agent.drift` in `data=synthetic_hierarchical` (empty in Stage 1):

```yaml
drift:
  learn_rate: {mode: linear, delta: 0.3}          # learn_rate rises over sessions
  biasL: {mode: toward_zero, frac: 0.8}           # |bias| shrinks toward 0
  softmax_inverse_temperature: {mode: multiplicative, rel: 0.5}  # sharper over time
```

Optionally add small `session_noise` per param. The ground-truth table already
records per-session (drifted) params + `session_frac`, so session-level recovery
is scorable out of the box.

## Planned model change

- `model.architecture.session_encoding_type`: `scalar` (or `fourier`), which turns
  on session conditioning (requires multisubject mode).
- Resolve `session_n_pretrain_steps` / `session_n_warmup_steps` (null → ~30% / ~20%
  of total steps) or set explicitly.
- `model.training.lambda_reg_session`: try `1.0` (zero-mean session-delta reg).

## Session-recovery scoring (Stage 2)

- Headline `likelihood_relative_to_groundtruth` as in Stage 1.
- Recovered **session-conditioning state** vs true per-session drift trajectory
  (correlate the model's session delta with `session_frac` and with the true
  per-session param deltas). Analysis script extended from the Stage-1
  model-agnostic recovery code.

## The baseline_rl role FLIPS at Stage 2 (key prediction)

At Stage 1 baseline_rl is the likelihood *ceiling* (~1.0) because it fits the true
static model. At Stage 2 the generator drifts within-subject, but baseline_rl still
fits ONE static parameter set per subject — it has no session axis. So it is now
**misspecified along the time axis** and must explain a moving policy with a
session-averaged compromise:

- **Prediction:** baseline_rl `likelihood_relative_to_groundtruth` drops **below 1.0**
  at Stage 2 (the more drift, the larger the drop).
- **The GRU with session conditioning ON should NOT degrade as much** — it can track
  the drift. The **gap that opens between GRU and static baseline_rl at Stage 2 = the
  value added by the session-conditioning mechanism.**
- This is a clean dissociation: GRU beats the static-per-subject RL fit *only* when
  there is real within-subject structure to capture → evidence the session
  embeddings encode the drift, not noise.

So at Stage 2 baseline_rl is the **static-model reference band the drift-aware model
is meant to exceed**, not a ceiling. Its degradation is the signal.

**Optional fair drift-aware baseline:** a per-session RL fit (or time-varying RL)
WOULD track the drift and not degrade — but then it is a different reference. Add it
as a SECOND Stage-2 reference if we want a drift-aware comparison point; keep the
static baseline_rl as the "no session mechanism" control.

## Compute / tracking

Same as gru-stage1. Project `embedding_recovery`, group `gru-stage2@<launch_id>`.
