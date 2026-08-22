
---

## Early-stopping + held-out scoring audit (2026-08-22)

Prompted by Han asking what the early-stopping rule is and whether online and
offline held-out scoring use the same criteria.

### The rule

`model.training.early_stopping`, monitoring **within-subject `eval_likelihood`**
at every checkpoint (`checkpoint_every_n_steps: 10000`):

| knob | value |
|---|---|
| `metric` | `eval_likelihood` (within-subject, NOT held-out) |
| `min_delta` | 0.003 |
| `patience` | 2 checkpoints with no new best |
| `overfit_guard` | 0.01 below best |
| `start_after_step` | 70000 (gate; unarmed before this) |
| `n_steps` (ceiling) | 150000 |

### Do both scoring paths use the same criteria? YES — verified

Both the online (end-of-train, `training_runner`) and offline
(`resume_heldout_beaker.py`) held-out evaluations call the SAME
`resolve_model_run(..., checkpoint_policy=...)` in
`post_training_analysis/generative_analysis.py`, and `best_eval` there is one
code path: `max(checkpoints, key=eval_likelihood)` over
`outputs/checkpoints/index.json`. The offline re-score reads
`checkpoint_policy` from the source run's own config, so it inherits the same
policy. The finetune knobs (`n_steps=500`, `lr=1e-3`) also match.

**So the 70k-vs-90k checkpoint difference I flagged earlier was NOT a scoring
inconsistency.** It is the rule behaving identically on two genuinely different
eval curves:

| run | argmax checkpoint | end value | shape |
|---|---|---|---|
| OFF seed0/1/2 | step_90000 / 80000 / 90000 | = max | **still climbing** |
| ON seed0/1/2 | step_50000 / 60000 / 60000 | below max | **peaked, then declined** |

The ON arm reaches its within-subject optimum ~30k steps earlier and then
overfits slightly; the OFF arm never stops improving within the budget. `best_eval`
correctly returns an earlier checkpoint for ON and the last one for OFF. Both
arms are scored **at their own within-subject optimum** — a defensible and
consistent criterion, and arguably the right one for a capacity-asymmetric
comparison. It is *not* a common step count, and any report must say which.

### But the rule is barely doing what it looks like it does

1. **`min_delta=0.003` is 10–15× larger than real late-training improvement.**
   Measured per-checkpoint deltas after 70k are ≤0.0003 (often 0.0000 or
   negative). So the "new best" test can essentially never pass once armed:
   `stale` increments almost every checkpoint regardless of whether the model is
   improving.
2. **The consequence is a deterministic stop, not an adaptive one.** Armed at
   70k with patience 2 and 10k checkpoints → stop at 90k. Observed: **14/18
   study-07 runs stop at ~90k**, the other 4 at ~80k (overfit guard). In
   `mice_data_scaling`, 49 runs stop at 91k. Early stopping is currently
   equivalent to a fixed ~90k-step budget with an overfit tripwire.
3. **It monitors the demoted metric.** The stop decision is driven by
   within-subject `eval_likelihood`, which this study treats as a diagnostic
   only (held-out-mouse LL is primary). `best_eval` likewise selects on
   within-subject. There is a `best_heldout` policy available but unused.

### Consequence for the reported effect

**The +0.0063 is not an artifact of the stopping rule.** The OFF arm's late slope
is +0.000156 per 10k steps, so closing a 0.0063 gap by training longer would take
**~405,000 additional steps** — 4.5× the entire budget, and the ON arm's own curve
is declining by then. The gap is not "OFF was stopped too early".

### Action

- Re-score BOTH arms through `resume_heldout_beaker.py` before publishing the
  curve, so every point comes from one code path (already the plan).
- Report the rule as a **fixed ~90k budget**, not adaptive early stopping.
- Consider `min_delta` ~0.0003 and/or `checkpoint_policy=best_heldout` for future
  variants — but NOT mid-study, since it would change what "best" means between
  cells.
