# Few-shot K=1 crash: does it survive mature-only? — 2026-06-23

Tests FUTURE_DIRECTIONS §8 / §1: the all-stage few-shot curve shows a **K=1 overfit crater**
— adapting the subject embedding on a *single* held-out session drops LL ~0.020 below
zero-shot (K0), recovering by K=4. Hypothesis: the crash is caused by adapting on a *naive
early-stage* session, so restricting eval to MATURE sessions (STAGE_FINAL/GRADUATED) should
remove it.

Method: 6 g6e mature groups (zeroshotM=K0, k1M=K1, k4M=K4 for v1/v2), per-held-out-subject
`eval_likelihood` from each run's `per_subject_likelihood` table, cohort-mean over subjects
(avg over 3 seeds first). Mature cohort n=117 (149→117; 32 mice lack mature sessions).

| cell | K0 (zero-shot) | K1 | K4 | **K1 dip (mature)** | K1 dip (all-stage) |
|---|---|---|---|---|---|
| v1_D10  | 0.7402 | 0.7279 | 0.7406 | **−0.0123** | −0.0217 |
| v1_D30  | 0.7431 | 0.7323 | 0.7435 | **−0.0108** | −0.0195 |
| v1_D100 | 0.7444 | 0.7324 | 0.7449 | **−0.0120** | −0.0201 |
| v1_D300 | 0.7449 | 0.7322 | 0.7453 | **−0.0127** | −0.0208 |
| v1_D614 | 0.7450 | 0.7324 | 0.7454 | **−0.0126** | −0.0209 |
| v2_D10  | 0.7402 | 0.7273 | 0.7404 | **−0.0129** | −0.0219 |
| v2_D30  | 0.7430 | 0.7297 | 0.7429 | **−0.0132** | −0.0220 |
| v2_D100 | 0.7453 | 0.7331 | 0.7452 | **−0.0122** | −0.0198 |
| v2_D300 | 0.7462 | 0.7355 | 0.7458 | **−0.0107** | −0.0163 |
| v2_D614 | 0.7464 | 0.7357 | 0.7465 | **−0.0107** | −0.0178 |

**Mean K1 dip: all-stage −0.0201 → mature −0.0120 = ~40% smaller, but still clearly negative.**

**Verdict: the K=1 crash PERSISTS on mature, attenuated ~40%.** So it is *partly* (~40%) the
naive-early-session effect the hypothesis predicted — adapting on an early/unstable session
hurts more — but the majority of the crater remains on well-behaved mature sessions. The
dominant cause is therefore **few-shot overfitting of the subject embedding to a single
session** (one session is too little data → the embedding fits that session's idiosyncrasies
at the expense of the rest), not the developmental stage of the adaptation session.

Supporting features (unchanged from all-stage):
- **Flat across D** (~10→614): the dip doesn't shrink with more pretraining mice → not a
  data-scaling / foundation-model signal, a protocol artifact of the 1-session finetune.
- **v1 ≈ v2**: session conditioning doesn't rescue it → not SC-related.
- **Full recovery by K=4** (K4 ≈ K0 in every cell, mature and all-stage): ≥4 sessions give
  the embedding enough signal to stop overfitting.

**Practical takeaway:** report zero-shot (K0) and K≥4, and treat K=1 adapted as an unreliable
operating point. Mature-only filtering improves but does not fix it — the fix is more
adaptation data (K≥4) or a regularized/early-stopped 1-session finetune.

Data: `mature_fewshot_curve.json`; figure `fig_mature_fewshot_curve.png` (solid=mature,
dashed=all-stage). W&B: groups `heldout-zeroshot-v{1,2}-zeroshotM@…182737/40`,
`heldout-rerun-v{1,2}-k{1,4}M@…182742/45/48/50` in
[mice_data_scaling](https://wandb.ai/AIND-disRNN/mice_data_scaling).
