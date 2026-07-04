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

## Compute / tracking

Same as gru-stage1. Project `embedding_recovery`, group `gru-stage2@<launch_id>`.
