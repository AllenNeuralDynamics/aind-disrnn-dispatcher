# Study 06 reports

One report per scan. Regenerated from committed data via the study Makefile.

| # | Report | Scan | Status |
|---|---|---|---|
| r1 | [Penalty selection](r1-penalty-selection.md) | existing 03+05 data (zero new compute) | ✅ live |
| r2 | Grid scaling curve | `mult-d-grid` (80 runs) | 🚀 launched 2026-07-18, awaiting runs |

## Planned content

- **r1 — penalty selection.** DONE. Kevin's in-sample-vs-held-out selection plot, built from
  existing study 03 (D=100) + study 05 (D=614) runs — zero new compute. Finding: β is free at
  D=100 but the generalization gap grows with D, and the held-out-optimal β is also the most
  overfit β. This is the motivation for r2's grid, and it changed r2's design (see below).
- **r2 — grid scaling curve.** The headline: held-out LL vs D×mult×β, overlaid on study 05's
  fixed-penalty curve and study 01's GRU curve. Launched as an 80-run grid (D×mult×β{3e-4,1e-3}×2
  seeds) directly motivated by r1 — see [notes.md](../../variants/mult-d-grid/notes.md) for the
  8 Beaker experiment IDs (payload-limit split) and W&B group `mult-d-grid@20260718-151409`.
- **(H2, follow-on)** generative switch-curve shape at the selected point — added with a
  `generative-*` rollout variant once r2 confirms which checkpoint(s) are worth rolling out.
