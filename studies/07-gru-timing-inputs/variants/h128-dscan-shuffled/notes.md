# h128-dscan-shuffled — shuffled-response control arm

Priority 3 of 4 in Han's ordering: all ON -> OFF no-new-inputs -> **OFF shuffled**
-> ON single-block (RT-only / licks-only).

## What this arm measures

The real arm widens the observation vector 2 -> 5, adding 9H parameters (+2.2% at
H128) and changing the total information-bottleneck / regularization budget. So a
win could in principle come from extra capacity rather than from the information
carried by reaction time and licking. This arm removes that ambiguity.

`data.timing_features.shuffle=true` permutes the raw response columns WITHIN each
`(subject_id, ses_idx)` group, applied after the attach and before encoding and
the previous-trial shift. Verified on real held-out loader output:

| property | real arm | shuffled arm |
|---|---|---|
| observation width | 5 | 5 |
| feature map | — | **identical** |
| channel means / sds | — | **identical to 4 dp** |
| corr(n_lick_left, choice) | −0.656 | −0.101 |
| corr(logRT, choice) | +0.197 | +0.172 |

So the arm is parameter-matched AND scale-matched, and the two contrasts are:

* `real − shuffled` = value of **trial-aligned** information
* `shuffled − OFF` = value of **session-level** response statistics alone

## This is a CONSERVATIVE control, not a zero-information baseline

Session-level structure survives by design. Session-demeaned correlations with
choice fall to ~0.003 (from −0.642 for `n_lick_left` and +0.030 for logRT), so
the residual raw correlation is between-session structure the shuffle
deliberately preserves — not trial alignment leaking through. Consequence:
`real − shuffled` is a **lower bound** on the information effect. State this when
reporting; a reader will otherwise assume the shuffled arm carries no signal.

## Seeding

`shuffle_seed` is left unset, which the wrapper resolves to the **run seed**. The
three replicates per D cell therefore see three *different* permutations. Had it
defaulted to a constant, the replicates would be three fits of one permutation and
the reported seed spread would measure optimizer noise only, understating
permutation variance. The held-out loader is instantiated independently, so it
receives the run seed explicitly too — otherwise the held-out frame would be
permuted with seed 0 while training used the run seed, a silent train/eval
mismatch (fixed in wrapper 7212d3d).

## Tier choice

`{priority: low, preemptible: true}` — deliberately the lowest tier. This arm sits
behind the timing-OFF arms that are still training and hold 15 slots. Low priority
fills only idle capacity and is evicted first, so queuing it now cannot slow the
OFF arms. The cost is wall-clock and repeated evictions, which autoResume absorbs;
that is the right trade for a control whose result is uninterpretable until the OFF
arms land anyway.

Both S3-capable H200 clusters are listed so placement stays orthogonal to the
swept D axis — fixing the ON arm's error of splitting the grid by D across
clusters, which correlated D with hardware.

## Grid

15 tasks: `data.subject_ratio` {0.016, 0.049, 0.163, 0.489, 1.0} x seed {0,1,2}.
All other knobs byte-identical to `h128-dscan`.

PRIMARY METRIC: `heldout/final/eval_likelihood`. Within-subject LL is a diagnostic
only (train-eval gap).

---

## First complete three-way cell: D≈10 (2026-08-22, 23:07 PT)

All three seeds of the shuffled arm at D≈10 finished. Held-out mouse likelihood:

| arm | mean | sd | seeds |
|---|---|---|---|
| no added inputs (OFF) | 0.72248 | 0.00166 | 3 |
| **shuffled control** | **0.71231** | 0.00422 | 3 |
| + RT & lick counts (ON) | 0.72388 | 0.00250 | 3 |

Decomposition:

| quantity | value | se |
|---|---|---|
| input-width cost (SHUF − OFF) | **−0.01017** | 0.00262 |
| trial-aligned information (ON − SHUF) | **+0.01157** | 0.00283 |
| net observed (ON − OFF) | +0.00140 | 0.00173 |

**The net effect hides two large opposing effects that nearly cancel.** The
information content is 8.3x the net, and the width cost cancels 88% of it. Had
we run only ON vs OFF at this cohort size we would have concluded "the features
barely help at D≈10" — a conclusion that is arithmetically true and
mechanistically wrong.

### The control validates itself

The concern with any negative control is whether it is really matched. The
train−eval gap says it is:

| arm | train − eval gap |
|---|---|
| OFF | +0.0229 |
| **SHUFFLED** | **+0.2734** |
| ON | +0.2375 |

The shuffled arm carries the same ~10x overfitting tax as the real arm (0.273 vs
0.238, both against 0.023 for OFF), confirming it is paying the capacity price
without receiving the information. That is exactly the design intent, measured
rather than assumed.

Note the shuffled arm is slightly WORSE than the real arm on this diagnostic
(+0.273 vs +0.238), consistent with real trial-aligned features being mildly
easier to fit than permuted ones — a second, independent sign that the
permutation removed something the model was using.

### Consequence for the headline

This changes the interpretation of the D curve's growth (see the study notes).
The ON−OFF gain rising +0.0014 → +0.0082 across D is now decomposable at least
at the small-D end: at D≈10 the information is already worth +0.0116, close to
the +0.0082 net seen at D≈614. So the leading hypothesis flips from

  "the information becomes more valuable as the cohort grows"

to

  "the information is worth roughly +0.010 throughout, and what changes with D
   is the vanishing width penalty."

CONFIRMATION PENDING: this rests on one D cell. The D≈30 and D≈100 shuffled cells
land in ~4h and ~2.5h; D≈614 in ~10h. If ON−SHUF is roughly flat across D while
SHUF−OFF shrinks toward zero, the reframing holds. If ON−SHUF itself grows, the
original "information scales" reading survives after all.
