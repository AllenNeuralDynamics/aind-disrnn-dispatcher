# Changelog — 05-disrnn-scaling-law

## 2026-07-13

- Study created to close the disRNN half of
  [issue #16](https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher/issues/16) (disRNN had
  never been trained above D=100).
- Three variants scaffolded: `smoke-d614` (full-cohort pipeline validation), `dscan-mult2` (the
  15-task scaling curve at study 03's mult=2 operating point), `mult-beta-d614` (12-task
  replication of study 03's mult×β grid at the full cohort).
- `smoke-d614` launched; the two grids are gated on it passing.
