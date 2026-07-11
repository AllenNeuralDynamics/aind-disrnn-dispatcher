# SC-stage verdict (mature-only eval, A) — 2026-06-23

Tests the hypothesis: SC's held-out benefit = accounting for curriculum/early-stage heterogeneity.
Method A (eval-only): re-run held-out adapted on MATURE sessions only (STAGE_FINAL/GRADUATED), using
the existing all-stage-trained v1/v2 checkpoints. Cohort 149→117 (32 mice lack mature sessions);
mature held-out LL is higher (~0.745 vs ~0.733 — mature behavior more predictable).
[First mature attempt was a NO-OP: a lone `mature_only` override is dropped unless a subject-selection
field is also set (`_selector_fields_present`); fixed by adding heldout_every_n/min_sessions. Verified
this run actually filtered (v1 LL 0.7426 ≠ all-stage 0.7285).]

| D | Δ(v2−v1) all-stage | Δ mature-only | Wilcoxon p (mature) |
|---|---|---|---|
| 10 | −0.00007 | −0.00007 | 1.5e-3 |
| 30 | −0.00010 | −0.00019 | 4.6e-6 |
| 100 | +0.00102 | +0.00073 | 4.5e-10 |
| 300 | +0.00135 | +0.00106 | 1.8e-15 |
| 614 | +0.00146 | +0.00116 | 3.1e-16 |

**Large-D (100/300/614) mean v2−v1: all-stage +0.00128 → mature +0.00098 = ~23% shrinkage.**

**Verdict:** PARTIALLY supports the hypothesis. ~1/4 of SC's large-D benefit was the early-stage
heterogeneity (vanishes mature-only) — the design rationale is real. But ~3/4 PERSISTS on mature
animals (p~1e-15), so SC mostly captures general session structure (within-mature drift / within-session
non-stationarity), not just training stage. SC is not "just a curriculum patch."

Caveat: eval-level test (A); models still trained all-stage. Definitive test B (retrain mature-only,
~30 runs) would say if SC is needed without early stages in TRAINING — but A already shows SC is mostly
not stage-driven, weakening the case for B.
