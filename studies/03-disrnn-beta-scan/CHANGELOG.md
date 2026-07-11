# Changelog — beta-scan (update-net-ratio, 100 mice)

Per-study log; newest first. One entry per merged PR or significant milestone.
Dates in America/Los_Angeles.

## 2026-07-11 — Study wrap-up + analysis normalization

- Normalized the study folder to `docs/study-organization.md` +
  `docs/posthoc-analysis.md`: added a single producer `analysis/beta_scan_report.py`
  (reads the committed clean grid, writes `beta_scan_summary.{json,csv}` +
  `fig_bottlenecks_by_mult.png` + `fig_mult_axis_heldout.png`, regenerates the
  report blocks via `analysis/update_reports.py`), `analysis/reports/`
  (`r1-bottleneck-sparsity-multiplier`, `r2-heldout-transfer`, `INDEX.md`),
  `analysis/provenance/`, `Makefile`, `environment.lock`, this changelog.
- Rebuilt the canonical clean grid `analysis/beta_scan_final_grid.csv` (52 finished
  runs = 43 main + 9 mult=10 supplement) directly from the W&B project after
  cleanup, carrying the full six-family threshold-free sparsity suite + held-out LL.
- Reused shared helpers from `../util/` (`_meta.py` with `study_root=`,
  `plot_style.py`) — not copied locally.
- **Metric correction (see `analysis/provenance/metric_caveat.md`).** All
  conclusions now use `total_openness` = Σ(1−σ); the scale-invariant
  `n_eff_open_frac` (which mis-ranked 19/43 runs and manufactured a false
  U-shape on the multiplier axis) is retained in the grid CSV but never headlined.
- Legacy exploratory `analysis/beta_scan_analysis.py` (old σ<0.1 `frac_open`
  metric) marked superseded by `beta_scan_report.py`.
- **W&B cleanup:** deleted 18 crashed/failed runs (all `heldout=None`, none holding
  unique data — every (mult,β) cell has 3–4 clean runs); project now 57 finished.
  Kept all main/supplement/ruleplot/restore runs. Audit:
  `analysis/provenance/launched.json`.

## 2026-07-10 — Rule-plot reruns

- Enabled `plot_choice_rule` / `plot_update_rules` by default (dispatcher
  `disrnn.yaml`); no-retrain restore reruns of the sparse best (`3dcb9217`) and
  least-sparse-by-openness (`45646c46`) runs to emit `fig/choice_rule` +
  `fig/update_rule_*` on W&B.

## 2026-07-06 — mult=10 supplement

- `updnet-ratio-100mice-mult10-supp` on free H200/L40S capacity (9 clean; 2
  deterministic NaN divergences at lr=5e-3/seed=0/β=3e-4; 1 OOM, retried to g6e).

## 2026-07-03 — Grid launched (staged short horizon)

- `updnet-ratio-100mice` variant: mult{1,2,5,10} × β{3e-4,1e-3,3e-3} × lr{1e-3,5e-3}
  × seed{0,1} = 48 tasks at 100 mice, 2-way, linear choice net.
- Staged short horizon: `n_steps` 150000→60000 with length bucketing (~1.86×
  step speedup) + trainer-agnostic W&B-artifact-restore "extend-later" path.
- Real-time threshold-free bottleneck-sparsity logging at loss pace.
- `smoke` variant proved length bucketing + resumability end-to-end.
- W&B project `disrnn_updnet_bottleneck_ratio_100mice`.
