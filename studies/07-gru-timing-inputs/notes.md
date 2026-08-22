
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

### Follow-up: is "fixed 90k budget" the same as "undertrained"? NO — it is D-dependent

Han asked whether the min_delta finding means many runs are undertrained. Measured
the within-subject eval curve of all 14 D-scan runs. The answer is that the D
sweep crosses **three different regimes**, so the fixed budget does not bias every
cell the same way:

| D (mice) | argmax checkpoint | late slope / 10k | regime |
|---|---|---|---|
| ≈10 | 10–20k | **−0.019 to −0.032** | **overtrained** — collapses to 0.52–0.56 by 80k |
| ≈30 | 30k | −0.011 to −0.016 | **overtrained** |
| ≈100 | 60–70k | −0.0003 to −0.0001 | **converged** (the bridge cell) |
| ≈300 | 90k (last) | +0.00017 to +0.00027 | **undertrained** — still rising when cut |
| ≈614 | 90k (last) | +0.00020 to +0.00050 | **undertrained** — still rising when cut |

So: **the small-D cells are overtrained, not undertrained** — they peak by 10–30k
and then fall off a cliff (the overfit guard is what stops them at 80k). Only the
**large-D end (D≈300, 614) is genuinely truncated** by the budget.

Note `best_eval` already protects the *reported number* in every regime: it
returns each run's own optimum, so the overtrained small-D cells are scored at
their 10–20k peak, not at their collapsed 80k value. The truncation only costs
what the large-D runs would have gained with more steps.

**Size of the large-D truncation.** A saturating fit was degenerate on monotone
data (it returned ~0 headroom, which is not credible), so bound it two ways from
the measured slope (+0.00026/10k averaged over the six large-D runs) over a
notional +50k steps:

* decaying increment (τ≈1 checkpoint): **+0.00015**
* slope held constant (optimistic): **+0.00131**

Against the +0.00630 timing effect that is 2–21% — small — but it is comparable to
or larger than the +0.00061 bridge offset, and critically it acts **only at the
top of the D axis**. So it distorts the **shape** of the scaling curve (flattening
the large-D end), which is exactly what a data-scaling study is trying to measure.

### Decision (Han, 2026-08-22): fix min_delta in FUTURE variants

Not mid-study — changing it now would make "best" mean different things in
different cells of the same grid. Planned for the next variant:

* `min_delta: 0.0003` (matched to the measured scale of late improvement, ~10x
  smaller than the current 0.003).
* Keep `start_after_step: 70000` and `patience: 2`; keep `overfit_guard: 0.01`
  (it is doing real work — it is what rescues the small-D cells).
* Raise the effective ceiling for large-D cells so a genuine "still rising" run
  can actually use it (`n_steps: 150000` is already set; the binding constraint
  is the stop rule, not the ceiling).
* Consider `checkpoint_policy: best_heldout` so selection and the primary metric
  agree — but note this changes what is being selected on, so it belongs in a
  variant that re-runs both arms, not a partial backfill.

Simulation caveat: at `min_delta` 0.001 and 0.0003 the *observed* d100-bridge runs
still stop at 90k, because their late deltas are ≤0.0003. The change bites for the
large-D cells whose slope is +0.0002-0.0005 per checkpoint, and for any future
cell that is still climbing. It is not a no-op, but it is not a large behavioural
change at D≈100 either.
