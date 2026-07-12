---
id: r5
slug: fewshot-adaptation
status: live
authors: [han]
wandb_groups:
  - heldout-rerun-*-fewshot-k*
  - heldout-rerun-*-mature2@*
inputs:
  # fewshot_curve.json has no committed producer (ad-hoc / manual aggregation).
  # mature_fewshot_curve.py produces the mature-only variant; rl_baseline.py overlays RL.
  script: ad-hoc
  data: analysis/fewshot_curve.json
  figure: analysis/fig_fewshot_curve.png
  related_scripts:
    - analysis/mature_fewshot_curve.py
    - analysis/rl_baseline.py
reproduce: python studies/data-scaling-law/analysis/rl_baseline.py  # refreshes RL overlay only
---

# Result 5 — few-shot adaptation curve (K = # new-mouse sessions used to adapt)

![few-shot curve](../fig_fewshot_curve.png)

| var | D | K=0 (zero) | K=1 | K=4 | K=full(~8) |
|---|---|---|---|---|---|
| v1 | 10 | 0.7264 | 0.7047 | 0.7252 | 0.7285 |
| v1 | 614 | 0.7315 | 0.7107 | 0.7307 | 0.7333 |
| v2 | 614 | 0.7331 | 0.7153 | 0.7326 | 0.7347 |
(all 10 cells show the same shape)

**Non-monotonic:** K=1 craters LL by ~−0.02 (a 4-dim embedding overfits one session at 500 steps/lr=1e-3), recovers to ≈zero-shot by K=4, only exceeds zero-shot at K=full (+~0.002). The dip is **D- and variant-independent** (protocol property, not scale). Caveat: largely a **protocol artifact** (no L2-to-mean / early-stop / lr-scaling for small K) — a tuned few-shot likely wouldn't crash. *(Mature-only few-shot in progress to test whether the crash is also driven by adapting on naive early-stage sessions.)* RL reference (per-subject mean **0.7211**) is shown on the figure for scale — even the K=1 crash floor (~0.71) is close to the RL band, and K=full sits ~+0.013 above it.

See also `mature_fewshot_verdict.md` (mature-only variant by `mature_fewshot_curve.py`).

## Related

- [[r4-zeroshot-vs-adapted]] — endpoints K=0 and K=full in tabular form.
- [[r6-sc-stage-mature-only]] — mature-only SC test (paired follow-up to this curve's mature variant).
