# Changelog — ignore-trials-scaling

Per-study log; newest first. One entry per merged PR or significant milestone.
Dates in America/Los_Angeles.

## 2026-07-11 — Study wrap-up + analysis normalization

- Normalized the study folder to `docs/study-organization.md` +
  `docs/posthoc-analysis.md`: added `analysis/` (`scaling.py`, `scaling.json`,
  `scaling.csv`, `fig_scaling.png`, `wandb_keys.py`, `reports/r1`,`r2` + `INDEX.md`,
  `provenance/`), `Makefile`, `environment.lock`, this changelog.
- Factored the provenance helper (`_meta.py`) and figure style/palette
  (`plot_style.py`) into shared `studies/util/`.
- **Correctness fix (see `analysis/provenance/backfill_history.md`).** The 12
  restore-backfilled cells had over-trained (wrong) held-out metrics; re-scored
  them exactly via the wrapper's new `resume_heldout_beaker.py`
  (commit `0a9141b`), reproducing native LR-engaged to <1e-6. `scaling.py` dedup
  corrected to prefer the native run over the over-trained restore run.
- Figure: SEM error bars (n=3 seeds/cell, noted in footer) + offset per-seed raw
  dots, presentation-grade fonts.

## 2026-07-03 — Grid launched

- `nxd-3way` variant: full 4×4×3 grid (H∈{16,64,128,256} × D≈{10,30,100,614} ×
  3 seeds = 48 tasks), `data.ignore_policy=include` (3-way L/R/ignore output).
- `validation-2way-vs-3way` variant: D≈10 smoke proving the 3-way pipeline
  end-to-end (ignore_policy ∈ {exclude, include}).
- W&B project `mice_ignore_scaling`; roadmap issue #23.
