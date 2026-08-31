# Changelog — 05-disrnn-scaling-law

## 2026-08-31

### Added
- `variants/generative-rl-baseline/extract_history_patterns.py` + three
  `rl_rollout_summaries/{ctt,bari,hattori}_history_patterns.json` (~123 KB each) — the frozen
  per-pattern `animal_mean` / `simulated_mean` (± SEM) rows for the history-dependent switch
  curve, streamed out of the 1.5 GB `history_dependent_switch_stats_no_figures.json` caches
  still on `/allen/aind/scratch/han.hou/tmp/rlgen/`. No re-simulation, no re-fit; each output
  is keyed by the sha256 of its source and the baseline's W&B run id, and the extractor asserts
  it reproduces the already-committed `quantitative_summary.subject_mean.abstract["3"]` values.

### Notes
- `variants/generative-rl-baseline/notes.md` — new section on why the pattern scatter was
  missing (the `_save_*_figures()` skip was aimed at the per-session panels but took the cheap
  pattern-comparison panel with it, and the committed summary keeps only per-pattern deltas,
  not the absolute coordinates) and what to persist instead next time.

## 2026-07-13

- Study created to close the disRNN half of
  [issue #16](https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher/issues/16) (disRNN had
  never been trained above D=100).
- Three variants scaffolded: `smoke-d614` (full-cohort pipeline validation), `dscan-mult2` (the
  15-task scaling curve at study 03's mult=2 operating point), `mult-beta-d614` (12-task
  replication of study 03's mult×β grid at the full cohort).
- `smoke-d614` launched; the two grids are gated on it passing.
