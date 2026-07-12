# v1 vs v2 — cell-level paired held-out test (2026-06-23)

**Question:** does session conditioning (v2, SC active) change held-out-mouse generalization
vs v1 (SC never engaged)? Cohorts are identical per (D, seed), so this is a matched-pair test.

**Result (15 matched (D,seed) pairs, `heldout/eval_likelihood`):**
- mean Δ(v2−v1) = **+0.00074** (sd 0.00073), **12/15 positive**
- paired t: t=3.93, **p=0.0015**; Wilcoxon: W=12, **p=0.0043** → significant.

**Effect grows with D (neutral at small D):**

| D | v1 | v2 | Δ |
|---|----|----|---|
| 10 | 0.7219 | 0.7218 | −0.0001 |
| 30 | 0.7250 | 0.7249 | −0.0001 |
| 100 | 0.7262 | 0.7273 | +0.0010 |
| 300 | 0.7267 | 0.7280 | +0.0014 |
| 614 | 0.7268 | 0.7282 | +0.0015 |

**Reading:** SC gives a small but real held-out gain that *increases with training-mice count*
(invisible unpaired — swamped by the ~0.005 across-D spread; pairing on matched cohorts
surfaces it). Magnitude is modest (~+0.0015 at full pool). Per-held-out-subject
repeated-measures (offline re-runs) will refine the per-mouse picture.
