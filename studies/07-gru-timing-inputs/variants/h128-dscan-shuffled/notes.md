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
