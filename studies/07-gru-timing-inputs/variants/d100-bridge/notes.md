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

---

## Results (2026-08-22)

### The bridge FAILED — study-01's H128 column is not reusable

| arm | held-out likelihood | seeds |
|---|---|---|
| study-01 `v2-sc-active` H128/D~100 | **0.72729** ± 0.00013 | 3 |
| this study, timing **OFF** | **0.72790** ± 0.00016 | 3 |

A **+0.00061 offset, ~4 SD apart**, with both bands internally tight — a real,
reproducible shift, not noise. Causes are confounded exactly as anticipated
before launch: newer wrapper SHA *and* newer image *and* different cluster
(aws-h200 vs onprem-h200).

Consequence: **paired timing-OFF arms are mandatory at every D.** The offset is
~10% of the effect being measured, so silently inheriting study-01's baseline
would have biased the whole curve. This is what the bridge was for.

### The effect at D≈100

| arm | held-out likelihood | seeds |
|---|---|---|
| timing OFF | 0.72790 ± 0.00016 | 3 |
| timing ON | **0.73420** ± 0.00043 | 3 |

**ON − OFF = +0.00630**, pooled sd 0.00032 → **19.5 sd**, t=23.9, p=1.8e-5, and
the arms do not overlap (min ON 0.73375 > max OFF 0.72808).

That is **83% of the logistic probe's +0.0076** and **1.1× the entire
D≈10→614 data-scaling gain (+0.0057)** — far larger than the 0.001–0.005 this
variant's own "Expectation" section predicted. The prediction was wrong in the
conservative direction: the GRU extracts *more* from RT/licking than a 3-lag
logistic probe suggested, not less.

### ⚠ The two arms were scored from DIFFERENT checkpoints

The ON numbers came from a **held-out-only re-score** (no retraining) after the
held-out bug below was fixed, and `checkpoint_policy=best_eval` selected
**step_70000**. The OFF arm's native end-of-train held-out used **step_90000**.
Both runs early-stopped at 90k (`eval_likelihood=0.7371 best=0.7378 stale=2/2`),
and the artifact for the ON runs contains both checkpoints — so this is genuine
`best_eval` behaviour on each arm's own eval curve, not a truncation artifact.
But it means the arms are **not scored at a common horizon**, and the ON arm was
scored at its own best while OFF was scored at its stopping point. The effect is
far too large (19.5 sd) to be explained by a 20k-step checkpoint difference on a
curve whose eval span is ~0.0007, but the comparison is not yet apples-to-apples
and the D-grid arms should be scored the same way.

### Held-out bug found and fixed (wrapper `35d6a19`)

The ON arm originally produced **no held-out metric at all**: the held-out bundle
is built by `heldout_finetuning.py` through its own loader path, which never
derived the timing columns, so the held-out tensor was 6 wide against a 9-wide
trained input and the restore failed with
`'multisubject_gru/~/gru/w_i' with retrieved shape (9, 384) does not match
shape=[6, 384]`. That warning is caught and logged, so the job exited 0 with the
study's primary metric silently absent. Fixed, then recovered without retraining
via `resume_heldout_beaker.py` (Beaker exp `01M0NAVPBNZGEX4DBH1M8BT7Y0`), which
logged `attached timing features -> 5 input feature(s)` and built a width-7
held-out tensor.

### DIAGNOSED: why study-01 does not reproduce (it is not code drift)

A field-by-field config diff against study-01 `v2-sc-active` H128/D~100 shows
**zero differences in `model.architecture` and `model.training`** — every
schedule, lr, batch, early-stop and conditioning knob matches. Two `data.*` keys
differ: `timing_features` (expected — it did not exist then) and **`snapshot`**:

| | study-01 (2026-06-22) | this study (2026-08-22) |
|---|---|---|
| `data.snapshot` | `None` | `20260603` |
| resolves to | `…/aind-dynamic-foraging-cache/session_table.parquet` | `…/snapshots/20260603/session_table.parquet` |

`snapshot=None` reads the **root table**, not the frozen snapshot. Those are
different objects: root has 24,865 sessions / 934 subjects (latest 2026-08-21),
pinned-20260603 has 23,868 / 902 (latest 2026-06-03).

Crucially — and this is where the naive "the root table has since rolled forward"
story is wrong — **every mouse study-01 used exists in the pinned snapshot**
(0 missing across all three runs). No mouse was unavailable. What differs is
smaller: **259 sessions across 32 subjects** were ingested between the snapshot
cut (Jun 3) and study-01's run (Jun 22) — 217 sessions for 24 subjects already in
the snapshot, plus 42 sessions belonging to 8 subjects that did not exist in it
at all (see the correction below).

**Correction (audit).** An earlier version of this note called all 32 of those
subjects "already-present", which is wrong and was caught in review. Joining the
window against the pinned subject list splits them:

| | subjects | sessions |
|---|---|---|
| already present in pinned 20260603 | 24 | 217 |
| **brand-new**, absent from pinned | **8** | **42** |

(Restricted to the mature/curriculum pool that actually drives ranking: 23
already-present / 179 sessions, and 5 new / 19 sessions.) The upper bound was
visible in the data all along — only 24 shared subjects have a differing session
count across the wider Jun3→Aug21 window, so 32 could never all be pre-existing.

This *strengthens* the mechanism rather than weakening it: 8 new subjects
entering the ranked pool displace ranks directly, on top of the 24 whose counts
grew.

That is enough, because cohort selection is **rank-based**: subjects are ordered
by session count, every 5th is reserved as held-out, and `subject_ratio` is
sampled from the remainder. Session counts near the selection boundary are
densely tied (19–27 subjects share each count in the 31–36 range), so one subject
gaining a session jumps a tie group and **cascades every rank below it**. Result
at matched `subject_sample_seed`: 1–3 mice swap per seed (seed 0 swaps 3, seeds
1 and 2 swap 1 each), and the **held-out population changes too** — so
"held-out likelihood" is being computed on partly different animals.

**So the +0.00061 offset is a cohort-composition difference, not evidence that
the wrapper changed behaviour.** Wrapper SHA, image and cluster are all still
confounded in principle, but they no longer need to be invoked: a sufficient
cause is identified and measured.

Two consequences:

1. **The headline effect is unaffected.** Verified directly: at every
   `subject_sample_seed`, the ON and OFF arms resolve to *identical* subject sets
   and both pin `snapshot=20260603` (seed 0: 99 mice, seeds 1–2: 101 mice each,
   `IDENTICAL=True` in all three). The +0.0063 is a within-cohort paired
   contrast, so selection drift cannot produce it.
2. **Paired OFF arms are still mandatory** across the D grid — not because the
   wrapper drifted, but because study-01's numbers are keyed to a different
   cohort draw and are not the right baseline for a pinned-snapshot study.

Lesson for the study conventions: **always pin `data.snapshot`.** An unpinned run
is not reproducible even with identical code, because the selection is
rank-sensitive to a table that grows underneath it.
