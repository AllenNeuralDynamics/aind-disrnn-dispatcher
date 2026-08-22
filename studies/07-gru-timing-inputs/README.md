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

Planned, not yet launched:

- **Paired timing-OFF arms** across the D grid — needed *only if* `d100-bridge`
  shows the OFF arm does not reproduce study-01's H128 band. If the bridge holds,
  study-01's H128 column is the baseline and this is not run.
- **Shuffled control** — the capacity-matched arm described above.

## Related

- `01-gru-scaling-law` — the baseline grid and the H128 column this study bridges to.
- `02-gru-scaling-law-ignore` — the other "more headroom" attempt, via a changed
  *output* rather than changed *inputs*.
