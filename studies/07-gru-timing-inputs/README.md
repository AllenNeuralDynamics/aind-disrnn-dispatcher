# Study: Timing inputs — previous-trial reaction time and lick counts

*Folder `07-gru-timing-inputs`. W&B project `mice_rt_lick_scaling`, one group per
variant. Filter runs by W&B **project**, not `meta.study`.*

**Question.** Studies 01 and 02 both found per-trial choice likelihood close to a
predictability ceiling on the *inputs we give the model* — previous choice and
previous reward. This study asks whether the ceiling is a property of the
**behaviour** or of the **input representation**: does telling the model what the
mouse *did* on the previous trial, beyond which side it chose, buy predictive
power that scales?

Two per-trial quantities are added as previous-trial inputs:

- **reaction time** — go-cue to first lick, log-encoded (it is ~log-normal,
  median ~0.14 s);
- **lick counts** — number of left and right licks in `[go-cue, go-cue + 2 s)`,
  fed as **raw per-side counts**, not total/difference.

**The only model/data change** vs study-01 is `data.timing_features.enabled:
false → true`. The wrapper derives the observation width from the input tensor,
so the input tensor widens **3 → 6** columns with no model-code or config-block
change. Verified in the `d100-bridge` job logs: the OFF arm builds
`x_names=['Subject ID', 'prev choice', 'prev reward']` (width 3) and the ON arm
`[..., 'prev log RT', 'prev n_lick_left', 'prev n_lick_right']` (width 6) — column
0 is the prepended multisubject index, so the observation count proper goes 2 → 5.
Quote the width, not `obs_size`: the GRU trainer sizes its input from
`xs.shape[2]` and has no `obs_size` parameter (that is a disRNN concept), so a
GRU study should not cite it. Everything
else (GRU H, scalar session conditioning, λ-forward schedule, lr, batch, held-out
cohort, snapshot pin) is held identical to `01-gru-scaling-law`'s `v2-sc-active` /
`nxd-grid` variants so the arms are comparable.

> **Inputs only — the output is unchanged.** `ys` remains the current trial's
> choice (`y_names=["choice"]`, `n_classes=2`). Unlike study 02, which changed the
> *output* to L/R/ignore and thereby moved the metric to a 1/3 chance baseline and
> a different trial support, this study leaves the metric identical to study-01's:
> per-trial P(choice) on the same engaged trials. Curves overlay study-01 directly,
> with no rescaling and no decomposition. A likelihood gain therefore means "prev
> RT/licking helps predict choice" — **not** that the model learned anything about
> how RT or licking are generated; nothing in the loss asks for that.

## Metric

**Primary: `heldout/final/eval_likelihood`** (held-out *mouse* generalization).
Within-subject likelihood is a **diagnostic only**, read as the train−eval gap to
tell which cells overfit — never a reported effect size. Measured on
`mice_data_scaling` at H=128, a 0.0076 effect is **10 sd** on held-out (median
per-cell seed sd 0.00076) but **1.7 sd** within-subject (sd 0.00455), and only
0.3 sd at D~10 (sd 0.0287); the two correlate just r=0.28 across the grid. This
also matters for the capacity caveat below, which lives entirely in the
within-subject metric.

## Effect size to expect

A session-held-out logistic probe over a 60-subject cohort gave **+0.0076**
normalized likelihood for RT + per-side licks over a 3-lag choice/reward
baseline. Treat that as a **loose upper bound, not a target**: it was measured
against a *logistic* model, and a GRU already extracts nonlinear and longer-range
history structure the 3-lag probe cannot. For scale, the entire data-scaling axis
(D~10 → 614 at H128) moves held-out likelihood only **+0.0057**, and the whole
H×D grid spans **0.0137** — so +0.0076 would exceed the total measured effect of
a 60× increase in mice. Expect **0.001–0.005, possibly smaller.**

## Known confound: capacity is not yet matched

The timing arm carries **9H = 1152 more parameters** at H128 (+2.2%: three gates
× three new inputs × H). The problem is not the magnitude but that its penalty is
**D-dependent** — measured train−eval gap at H128 is +0.030 at D~10, +0.004 at
D~30, −0.003 at D~100. So a higher-capacity arm can look *worse* at small D and
overtake as D grows, mimicking exactly the "information that scales" signature
this study tests for.

The fix is a **shuffled-feature control**: RT/lick values with identical marginals
but destroyed trial alignment, which is parameter- *and* scale-matched at every D,
making `timing − shuffled` attributable to information alone. It is deferred to a
later variant by decision; until it runs, a positive result is not fully
separated from the capacity advantage. State this in every report.

## Implementation caveats

- **Update-rule plots are unavailable** for >2 inputs — upstream
  `plotting.plot_update_rules` hardcodes exactly two observations. Training and
  evaluation are unaffected (the wrapper guards the call), but any variant with
  timing on loses those figures.
- **Continuous inputs required a dtype fix.** Upstream
  `aind_disrnn_utils.create_disrnn_dataset` allocates `xs` as int64 via
  `np.full(..., -1)`, truncating float columns (it collapsed 24,149 distinct
  log-RT values to 7). The wrapper routes continuous-feature datasets through a
  float-safe builder; integer-only runs still call upstream, so prior studies stay
  bit-for-bit reproducible.
- **Standardization is on by default** and matters for the disRNN, whose
  bottleneck KL is quadratic in input magnitude (raw lick counts entered at mean
  `mus²` = 32 vs 0.37 for a binary channel). Constants are fixed and global — not
  per-run fitted (train and held-out loaders are instantiated independently) and
  never per-subject (that would erase the between-mouse variation the subject
  embedding exists to capture).

## Variants index

| Variant | What differs | Status | W&B group | Beaker exp |
|---|---|---|---|---|
| [`d100-bridge`](variants/d100-bridge/) | One D cell (D~100), timing OFF vs ON × 3 seeds. OFF arm doubles as a bridge control against study-01's H128/D~100 held-out (0.72729 ± 0.00013) on a newer wrapper SHA, image and cluster. | launched 2026-08-22 | `d100-bridge@<launch_id>` | see `launch_record/` |

| [`h128-dscan`](variants/h128-dscan/) | Timing arm ONLY across the D grid: `subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}` × 3 seeds, H128. The study's headline curve. Launched in parallel with `d100-bridge` since the timing arm is needed at every D either way. | launched 2026-08-22 | `h128-dscan@20260821-231301` | `01M0M1KQQCSYZMRE9S9F5SSZV5` |

| [`h128-dscan-onprem`](variants/h128-dscan-onprem/) | The same timing arm for `subject_ratio ∈ {0.163, 0.489, 1.0}` × 3 seeds, moved to `octo-hub-onprem-h200` because aws-h200 had saturated with this study's own tasks. Identical recipe. | launched 2026-08-22 | `h128-dscan-onprem@20260822-012657` | `01M0M997B9XNCY0W1Z9FK69GA8` |

> **Cluster note.** GCP clusters (`octo-hub-gcp-h100`, `octo.hub-gcp-h200`) cannot
> reach the AWS S3 parquet cache and are unusable for any DB-backed run here, however
> many GPUs they show free. Usable S3-backed targets: `octo-hub-aws-h200`,
> `octo-hub-onprem-h200`, `octo-hub-aws-l40s`, `aipbd-aws-h200`. Also note the D grid
> is split across two clusters (see `h128-dscan-onprem` notes), so D and cluster are
> correlated — both are H200 and study-01's H128 column trained on onprem, but a
> cluster-level artifact would alias onto the D axis.

| [`h128-dscan-off-bigD`](variants/h128-dscan-off-bigD/) | Paired timing-OFF baseline for the large cohorts (`subject_ratio ∈ {0.163, 0.489, 1.0}` × 3 seeds), allocated/non-preemptible tier. Became mandatory because the bridge showed the OFF arm does *not* reproduce study-01's H128 band (cohort draw differs under the pinned snapshot). | done | `h128-dscan-off-bigD@<launch_id>` | `01M0NDR91DA7H79G71B1HVHYAR` |
| [`h128-dscan-off-smallD`](variants/h128-dscan-off-smallD/) | Paired timing-OFF baseline for the small cohorts (`subject_ratio ∈ {0.016, 0.049}` × 3 seeds), unallocated/high-priority preemptible tier. | done | `h128-dscan-off-smallD@<launch_id>` | `01M0NDS1JYZ58ZC2EHGK5X0K4S` |
| [`h128-dscan-shuffled`](variants/h128-dscan-shuffled/) | The capacity-matched **shuffled control**: ON inputs permuted within session (marginals + between-session structure preserved, trial alignment destroyed). `ON − shuffled` isolates information from the input-width cost. | done | `h128-dscan-shuffled@<launch_id>` | `01M0NSRWBVWQENPPNDTG4NKX28` |
| [`h128-dscan-rt-only`](variants/h128-dscan-rt-only/) | Single-block arm: previous-trial **reaction time only** (1 added channel) × D grid × 3 seeds. Decomposes the combined effect. | done | `h128-dscan-rt-only@<launch_id>` | `01M0QQC7WX5NTKDKJV50SM99HQ` |
| [`h128-dscan-licks-only`](variants/h128-dscan-licks-only/) | Single-block arm: previous-trial **lick counts only** (2 added channels) × D grid × 3 seeds. | done | `h128-dscan-licks-only@<launch_id>` | `01M0QQCGWF69Z04QJ6PR8PKCRX` |

All planned arms have run. The `d100-bridge` OFF arm did **not** reproduce
study-01's H128/D≈100 band (the pinned-snapshot cohort draw differs), so the
paired OFF arms above were run at every D rather than inheriting study-01's
column — see the Verdict.

## Verdict

Full tables and figures: [`analysis/reports/`](analysis/reports/INDEX.md). Headline
figures at the study root: `fig_block_decomposition.png`, `fig_threeway.png`,
`fig_selection_breakdown.png`. All numbers are held-out **mouse** likelihood at
H=128, 3 seeds/cell, and hold at D≥100 (D≤30 excluded — see below).

**1. Previous-trial response features help, and the effect is real but FLAT.**
Adding previous-trial reaction time + lick counts raises held-out likelihood by a
constant **~+0.0075** across the trustworthy cohort range (D≈100→614). The raw
ON−OFF curve *appears* to grow with cohort size (+0.0014 at D≈10 → +0.0082 at
D≈614), but the shuffled control shows that growth is a **vanishing input-width
overfitting penalty** (shuffled−OFF rises from −0.010 to ~0), not information
scaling. ON−shuffled — information alone, parameter- and scale-matched — is flat.
Without the control the naive reading ("the feature matters more with more data")
is wrong.

**2. Lick counts carry ~all of it; reaction time is small but real and growing.**
At D≥100, licks-only recovers **~98%** of the combined benefit; RT-only ~7%. The
counter-intuitive part: reaction time is near-*orthogonal* to the existing
(prev-choice, prev-reward) inputs yet contributes little, while lick counts are
heavily *redundant* with prev choice yet carry the effect — novel variance is not
useful variance. RT's share is statistically distinguishable from zero (pooled
D≥100, p≈0.004) and, unlike licking, still growing at D=614, suggesting it is the
block that would benefit from a larger cohort. The two blocks are mildly
**sub-additive** (redundant with each other), not synergistic.

**3. Below D≈100 the results are untrustworthy — a measurement artifact, not
behaviour.** Held-out scoring uses the best-within-subject checkpoint
(`best_eval`). At small cohorts the within-subject signal it selects on has itself
collapsed: corr(within-subject, held-out) across seeds is ≈0 at D≈10 and rises to
1.0 at D≈614, and seed noise there dwarfs the +0.0075 effect. So `best_eval`
cannot pick a good checkpoint and the D≤30 cells are excluded from every estimate.
A future variant should select on held-out LL (`best_heldout`); this ties into
the early-stopping `min_delta` follow-up (`min_delta` is ~10× too large to fire,
making the ~90k-step budget effectively fixed rather than adaptive).

**4. Study-01's H128 column is not a reusable baseline.** The `d100-bridge` OFF
arm sits +0.0006 (≈4 SD) above study-01's H128/D≈100 band. The cause is not code
drift (a field-by-field config diff is identical) but `data.snapshot`: study-01
ran unpinned while this study pins `20260603`, and cohort selection is rank-based
over densely-tied session counts, so a modest difference in ingested sessions
reshuffles which mice land in the train/held-out split. Hence the paired OFF arms
were run at every D. **Convention adopted: always pin `data.snapshot`** — an
unpinned run is not reproducible even with byte-identical code.

## Related

- `01-gru-scaling-law` — the baseline grid and the H128 column this study bridges to.
- `02-gru-scaling-law-ignore` — the other "more headroom" attempt, via a changed
  *output* rather than changed *inputs*.
