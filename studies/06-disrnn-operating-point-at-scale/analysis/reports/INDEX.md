# Study 06 reports

One report per scan. Regenerated from committed data via the study Makefile.

| # | Report | Scan | Status |
|---|---|---|---|
| r1 | [Penalty selection](r1-penalty-selection.md) | existing 03+05 data (zero new compute) | ✅ live |
| r2 | [Scaling surface (live)](r2-scaling-surface.md) | `mult-d-grid` (80 runs) | 🔴 live — regenerates as the grid progresses (`make pull && make r2`) |

## Planned content

- **r1 — penalty selection.** DONE. Kevin's in-sample-vs-held-out selection plot, built from
  existing study 03 (D=100) + study 05 (D=614) runs — zero new compute. Finding: β is free at
  D=100 but the generalization gap grows with D, and the held-out-optimal β is also the most
  overfit β. This is the motivation for r2's grid, and it changed r2's design (see below).
- **r2 — grid scaling curve.** The headline: held-out LL vs D×mult×β, overlaid on study 05's
  fixed-penalty curve and study 01's GRU curve. Launched as an 80-run grid (D×mult×β{3e-4,1e-3}×2
  seeds) directly motivated by r1 — see [notes.md](../../variants/mult-d-grid/notes.md) for the
  full Beaker experiment history and W&B group `mult-d-grid@20260718-151409`. That history grew
  to **13 experiments** across 5 launch records (`launch_record/beaker_*.json`): the original 8
  (payload-limit split), +2 resubmitting 20 tasks lost to a bad node, +1 NaN determinism probe,
  +1 rescue off a second bad node, +1 tier-1 relaunch of the final 3 cells. **`grid.csv` also
  contains 6 held-out values recovered post-hoc** (flagged `heldout_backfilled`) — see notes.md
  before using the data.
- **(H2, follow-on)** generative switch-curve shape at the selected point — added with a
  `generative-*` rollout variant once r2 confirms which checkpoint(s) are worth rolling out.
