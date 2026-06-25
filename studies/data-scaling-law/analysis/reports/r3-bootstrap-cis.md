---
id: r3
slug: bootstrap-cis
status: live
authors: [han]
wandb_groups:
  - heldout-rerun-v1@*
  - heldout-rerun-v2@*
inputs:
  script: analysis/bootstrap_scaling.py
  data: analysis/bootstrap_scaling.json
  figure: null
reproduce: python studies/data-scaling-law/analysis/bootstrap_scaling.py
---

# Result 3 — bootstrap CIs on the scaling shape (resample 149 held-out mice ×1000)

Per-mouse-mean LL (equal-weight; differs from the trial-weighted Result 1 levels). Within-cohort increments are tight even though absolute per-D levels aren't (mice vary in predictability).

| quantity | v1 | v2 |
|---|---|---|
| frac of total gain by D=100 | 0.90 [0.89, 0.91] | 0.85 [0.84, 0.87] |
| late gain D=100→614 | +0.00049 [+0.00042, +0.00056] | +0.00092 [+0.00084, +0.00100] |

Both late-gain CIs **exclude 0** → not perfectly saturated; a small real slope persists (≈2× larger under SC). A saturating fit `L = L∞ − A·D^−α` (seed-averaged) gives **v1: L∞=0.727, α=0.88** (asymptote essentially reached by D=614) and **v2: L∞=0.729, α=0.52** — SC's shallower exponent means it keeps rising, with the asymptote ~+0.001 above the observed D=614, echoing v2's ~2× late slope. **~85–90% of the data benefit is captured by ~100 mice.** The residual is small per-trial but statistically real and, per the per-session compounding (below), genuine evidence — just headroom-poor on this metric.

## Related

- [[r1-heldout-scaling-curve]] — cell-level aggregate that this CIs the shape of.
- [[r2-per-mouse-repeated-measures]] — per-mouse pairing analysis.
- [[r7-nxd-joint-scaling-grid]] — joint N × D scan that supersedes/extends the 1-D scaling shape.
