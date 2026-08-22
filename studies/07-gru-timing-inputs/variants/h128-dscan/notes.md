# Variant h128-dscan — timing arm across the D grid (launched 2026-08-22)

**Goal:** the study's headline curve. Does the predictive gain from previous-trial
RT + lick counts **grow, saturate, or wash out** as the training cohort grows?

Grid: H=128, `data.timing_features.enabled=true` fixed,
`data.subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}` (D≈10/30/100/300/614)
× seeds {0,1,2} = **15 resumable tasks**.
Beaker exp `01M0M1KQQCSYZMRE9S9F5SSZV5`, W&B group `h128-dscan@<launch_id>`.

## Config parity

Verified by diffing the sweep command block against study-01
`v2-sc-active` (the source of study-01's H128 column). The **only** deltas:

| delta | why |
|---|---|
| `+ data.timing_features.enabled=true` | the science knob |
| `wandb.project mice_data_scaling → mice_rt_lick_scaling` | one project per study |

Nothing from study-01 is missing. Same λ-forward schedule, `n_steps=150000`,
`lr=1e-5`, gated early stop at 70k, `length_bucketing`, snapshot pin.

## Why this launched before the d100-bridge verdict

The timing arm is required at **every** D regardless of how the bridge lands. The
bridge decides only where the *baseline* comes from:

- bridge holds → reuse study-01's H128 column, no OFF arms needed;
- bridge fails → launch paired timing-OFF arms as a separate variant.

Neither outcome invalidates anything here, so serializing would have burned ~12 h
of idle wall-clock for no information. Capacity was available (10 free slots on
`aws-h200`, empty queue).

## The D≈100 duplicate is deliberate

`subject_ratio=0.163` appears both here and in `d100-bridge`'s ON arm. Keeping
every point of the headline curve inside one launch avoids an un-diagnosable kink
from pooling across launches, and the duplicate is a free cross-launch
seed-level reproducibility check at one cell.

## Metric and expectation

Primary: **`heldout/final/eval_likelihood`**. Within-subject LL is a diagnostic
only (train−eval gap). Expect a realized gain of **0.001–0.005**, *not* the
+0.0076 logistic probe — that was measured against a 3-lag logistic rather than a
GRU, and it exceeds the total held-out movement of the entire D≈10→614 axis
(+0.0057).

## Carried caveat: capacity is not matched

The timing arm has 9H = 1152 more parameters (+2.2%) and the penalty is
**D-dependent** (train−eval gap at H128: +0.030 / +0.004 / −0.003 at D≈10/30/100),
so it distorts curve *shape*. Read D≈10 as a separate overfitting regime. The
shuffled-feature control that fixes this is deferred by decision — state it in
every report until it runs.
