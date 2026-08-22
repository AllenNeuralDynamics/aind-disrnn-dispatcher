# Variant d100-bridge — one D cell, timing OFF vs ON (launched 2026-08-22)

**Goal:** two things at once, at a single cohort size before spending the full grid.

1. **Bridge:** does a `timing_features.enabled=false` run on the NEW wrapper SHA
   reproduce study-01's H128/D~100 held-out likelihood? If yes, study-01's H128
   column can serve as the baseline for the D grid instead of re-running it.
2. **First read of the effect:** timing ON minus timing OFF at D~100, judged
   against per-cell seed spread.

## Design — study-01 v2-sc-active, one knob changed

Every knob in `sweep.yaml` is copied verbatim from
`studies/01-gru-scaling-law/variants/v2-sc-active/sweep.yaml` (the source of
study-01's H128 column): scalar session conditioning, λ-forward (pretrain 30k,
warmup 30k→50k ⇒ full λ at 50k), `n_steps=150000`, `lr=1e-5`,
`checkpoint_run_heldout_eval=false`, `length_bucketing=true`, gated early stop at
70k. The only additions:

- `data.timing_features.enabled ∈ {false, true}` — the science knob.
- `data.subject_ratio=0.163` pinned (D~100) — single cell for this variant.

Grid: 2 arms × seeds {0,1,2} = **6 tasks**.

## Bridge target (fixed before launch)

Study-01 group `v2-sc-active@20260622-144622`, H=128, D~100, 3 runs:

| run | D | held-out |
|---|---|---|
| `v2-sc-active-20260622-144622-4becb8b9` | 99 | 0.72726 |
| `v2-sc-active-20260622-144622-35610e21` | 101 | 0.72717 |
| `v2-sc-active-20260622-144622-e7f3851f` | 101 | 0.72743 |

**mean 0.72729, sd 0.00013.** The OFF arm landing inside that band means the
bridge holds. Note study-01 used `WRAPPER_REF=65c3350d`; we run a newer SHA
(float-safe dataset builder + update-rule plotting guard) on a different cluster
and image, so this is a genuine re-measurement, not a formality.

## Metric

**Primary: `heldout/final/eval_likelihood`.** Within-subject likelihood is a
**diagnostic only** (via the train−eval gap), never a headline number — measured
on `mice_data_scaling` at H=128, a 0.0076 effect is 10 sd on held-out (median
per-cell seed sd 0.00076) but 1.7 sd within-subject (sd 0.00455), and only 0.3 sd
at D~10 (sd 0.0287). The two metrics correlate just r=0.28 across the grid.

## Expectation

Deliberately lower than the logistic probe's `+0.0076`. That probe was measured
against a **3-lag logistic** baseline, not a GRU; the GRU already captures
nonlinear and longer-range choice/reward history the logistic could not. For
scale, the entire data-scaling axis (D~10→614 at H128) moves held-out only
+0.0057, and the whole H×D grid spans 0.0137 — so +0.0076 would exceed the total
measured effect of a 60× increase in mice, which is implausible. Expect
**0.001–0.005, possibly smaller.**

## Known caveats carried into this variant

- **Capacity is not matched.** The timing arm has 9H = 1152 more parameters
  (+2.2% at H128). The penalty for extra capacity is *D-dependent* (train−eval
  gap at H128: +0.030 at D~10, +0.004 at D~30, −0.003 at D~100), so it distorts
  curve *shape*, not just level. A shuffled-feature control arm — same marginals,
  destroyed trial alignment, parameter- and scale-matched at every D — is the fix
  and is deferred to a later variant by decision. Until it runs, a positive
  result is not fully separated from the capacity advantage. At D~100
  specifically the gap is ≈0, which is part of why this cell was chosen first.
- **Update-rule plots are unavailable** with >2 inputs (upstream
  `plotting.plot_update_rules` hardcodes exactly two observations). Training and
  evaluation are unaffected; the wrapper now guards the call instead of dying.

## Launch history

- Two earlier submissions (`01M0KXMD7WR0DBBK7HN4GWDE6M`,
  `01M0KXVN5KKMCWKC89MNYHJTN2`) were launched from `/tmp` and are **abandoned**:
  they stamped `meta.study=adhoc, meta.variant=sweep` (the launcher derives both
  from the `studies/<study>/variants/<variant>/` path), and their on-prem cluster
  could not pull the image. Both stopped; superseded by the launch from this
  folder. See `launch_record/` for the intervention record.
