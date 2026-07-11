---
aliases:
  - ignore-trials reports index
  - ignore-scaling results index
tags:
  - index
  - report
  - ignore-trials-scaling
---

# Ignore-trials-scaling study — results index

Study cover (question, verdict, provenance): [../../README.md](../../README.md).

| id | slug | status | one-line headline |
|---|---|---|---|
| [r1](r1-lr-engaged-scaling.md) | lr-engaged-scaling | live | 3-way head is free on choice: L/R-engaged matches the 2-way ceiling and scales with D (best H256/D614 = 0.7315, still climbing) |
| [r2](r2-ignore-detection-scaling.md) | ignore-detection-scaling | live | ignore-class detection PR-AUC ~0.61→0.64 with D (far above ~0.05–0.10 base rate); recall capped ~0.47 regardless of scale |

## Conventions

Both reports are produced by a single script (`analysis/scaling.py`, which pulls
the live W&B grid, writes `analysis/scaling.json` + `fig_scaling.png`, and calls
`analysis/update_reports.py` to regenerate the `<!-- BEGIN result-N -->` /
`<!-- END result-N -->` blocks). Prose outside those markers is human-edited.
Regenerate with `make -C studies/ignore-trials-scaling` (needs `WANDB_API_KEY`).
See [[posthoc-analysis]] for the full contract.

**Metric caveat (both reports).** The raw 3-way normalized likelihood (chance
1/3) is **not** comparable to the 2-way L/R likelihood (chance 1/2) by
subtraction. r1 scores the conditional L/R likelihood on engaged trials (the
like-for-like comparator); r2 scores ignore-class detection separately.
