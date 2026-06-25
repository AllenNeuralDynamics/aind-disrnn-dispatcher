---
id: r6
slug: sc-stage-mature-only
status: live
authors: [han]
wandb_groups:
  - heldout-rerun-*-mature2@*
inputs:
  # mature_sc_verdict.json/.md have no committed producer (ad-hoc / manual aggregation).
  script: ad-hoc
  data: analysis/mature_sc_verdict.json
  figure: null
  notes: see also analysis/mature_sc_verdict.md (manually authored alongside the JSON)
reproduce: TBD  # producer script not yet wired
---

# Result 6 — SC-stage verdict (mature-only eval)

Tests "is SC's benefit just accounting for curriculum/early-stage heterogeneity?" Re-ran held-out *adapted* on **mature sessions only** (STAGE_FINAL/GRADUATED) using the same all-stage-trained checkpoints (cohort 149→117; mature LL higher, ~0.745, as mature behavior is more predictable).

| D | Δ(v2−v1) all-stage | Δ mature-only |
|---|---|---|
| 100 | +0.00102 | +0.00073 |
| 300 | +0.00135 | +0.00106 |
| 614 | +0.00146 | +0.00116 |

Large-D mean v2−v1: **+0.00128 (all-stage) → +0.00098 (mature) = ~23% shrinkage** (mature still p~1e-15). **So ~¼ of SC's benefit was the early-stage heterogeneity (the design rationale was partly real), but ~¾ persists on mature animals → SC mostly captures general session structure (within-mature drift / within-session non-stationarity), not just training stage.** (Eval-level test; models still trained all-stage. A definitive "retrain mature-only" test is deprioritized given this.)

See also `mature_sc_verdict.md` (committed alongside the JSON) and `mature_fewshot_verdict.md` (mature-only few-shot follow-up by `mature_fewshot_curve.py`).

## Related

- [[r1-heldout-scaling-curve]] — all-stage SC effect that this re-tests on mature-only.
- [[r5-fewshot-adaptation]] — mature-only variant follow-up.
