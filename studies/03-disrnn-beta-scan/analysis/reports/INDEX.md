---
aliases:
  - beta-scan reports index
  - updnet-ratio beta-scan results index
tags:
  - index
  - report
  - beta-scan
---

# β-scan (update-net-ratio, 100 mice) study — results index

Study cover (question, verdict, provenance): [../../README.md](../../README.md).

| id | slug | status | one-line headline |
|---|---|---|---|
| [r1](r1-bottleneck-sparsity-multiplier.md) | bottleneck-sparsity-multiplier | live | multiplier monotonically closes the interaction bottleneck (Σ(1−σ) 3.11→0.00 at weak β); model compensates by opening update←subject / choice←latent |
| [r2](r2-heldout-transfer.md) | heldout-transfer | live | held-out transfer is flat across the multiplier (~0.008 LL full range); set by base β, not multiplier — sparsification is "free" |

## Conventions

Both reports are produced by a single script (`analysis/beta_scan_report.py`,
which reads the committed clean grid `analysis/beta_scan_final_grid.csv`, writes
`analysis/beta_scan_summary.{json,csv}` + the two figures, and calls
`analysis/update_reports.py` to regenerate the `<!-- BEGIN result-N -->` /
`<!-- END result-N -->` blocks). Prose outside those markers is human-edited.
Regenerate with `make -C studies/beta-scan`. See [[posthoc-analysis]] for the
full contract.

## Metric caveat (both reports)

Openness is reported as **`total_openness` = Σ(1−σ)** over a bottleneck family's
channels — the absolute open capacity (~0 = fully closed). We deliberately do
**not** headline **`n_eff_open_frac`** (the normalized participation ratio):
it is scale-invariant, so it reports a spuriously high value even when a
bottleneck is fully shut (it measures how the vanishing residual weights are
*distributed*, not how much openness exists). On this grid `n_eff_open_frac`
mis-ranked 19/43 runs and manufactured a false U-shape on the multiplier axis;
`total_openness` shows the true monotone closure. The full threshold-free suite
(including `n_eff_open_frac`) is kept in `beta_scan_final_grid.csv` for reference.
