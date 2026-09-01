# h128-dscan-licks-only — lick-counts-only arm

Experiment `01M0QQCGWF69Z04QJ6PR8PKCRX` (15 tasks: subject_ratio {0.016, 0.049,
0.163, 0.489, 1.0} x seed {0,1,2}), W&B group `h128-dscan-licks-only`, low priority
+ preemptible on both S3-capable H200 clusters.

Priority 4 of 4. Knobs byte-identical to `h128-dscan`; only
`data.timing_features.reaction_time=false` / `lick_counts=true` differ. Both
knobs were already declared in all three dispatcher data configs, so unlike the
shuffled launch no config change was needed — checked before editing.

## What this arm settles

The three-way grid established that the combined arm carries **+0.0075 of
trial-aligned information, flat across cohort size**. This arm and
`h128-dscan-licks-only` ask which feature block carries it. The blocks are not
symmetric:

* log RT is near-orthogonal to the existing inputs (R² ≈ 0.00 on prev
  choice + reward) but contributed only +0.0009 of the logistic probe's +0.0067.
* lick counts are heavily redundant with prev choice (R² ≈ 0.61–0.63) yet carried
  +0.0061 of it.

So a large licks-only effect would mean the value lies in a **nonlinear read of a
partly-redundant channel**, not in new information per se.

## 3 seeds is a FIRST LOOK, not a null test

Han chose 3 seeds for both arms. Scaling the combined GRU effect by the probe
split predicts RT-only ≈ +0.0008, against a measured held-out seed sd of 0.00038
at D≈100 — roughly 2 se at n=3. **If RT-only lands within ~2 se of zero that is
not evidence of no effect**; add seeds before drawing any conclusion. The
licks-only arm (predicted ≈ +0.005) is well powered at n=3.

## Width caveat — do not compare directly to the combined arm at small D

This arm adds **2** input channels; RT-only adds 1; the combined arm adds 3.
The three-way grid measured the width cost at −0.010 (D≈10), −0.001 (D≈100), and
≈0 (D≥300). So at small cohorts these arms pay a *smaller* width penalty than the
combined arm and are not directly comparable to it there. **Compare each against
OFF at matched D, and prefer the D≥100 cells** for the decomposition.

## Second readout: additivity

If (RT-only) + (licks-only) ≈ combined, the blocks contribute independently. A
shortfall means they interact — plausible, since the mechanism analysis found RT
predicts switching independent of reward, while lick counts predict it only after
an unrewarded trial.
