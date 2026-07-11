---
aliases:
  - embedding-recovery reports index
tags:
  - index
  - report
  - embedding-recovery
---

# 04-gru-vs-disrnn-embedding-recovery — results index

Study cover (question, verdict, provenance): [../../README.md](../../README.md).

| id | slug | status | one-line headline |
|---|---|---|---|
| [r1](r1-gru-ladder.md) | gru-ladder | live | GRU embedding recovers true structure (97.5–100%) exactly where the correct-model baseline flips (S2b 0.94, S3 47%, S4a 70%); embedding size is the identifiability knob |
| [r2](r2-disrnn-replication.md) | disrnn-replication | live | interpretable disRNN decodes family at 0.95–0.98 (scalar) / 0.75–0.90 (none), nearly matching GRU's 100%, at a ~4–6 pt likelihood cost |

## Conventions

Both reports are produced by a single script (`analysis/recovery_report.py`),
which reads the committed curated grids (`analysis/ladder_results.csv`,
`analysis/stage4b_recovery_grid.csv`, `analysis/disrnn_stage4a_recovery_grid.csv`),
writes `analysis/recovery_summary.{json,csv}` + three figures, and regenerates the
`<!-- BEGIN result-N -->` / `<!-- END result-N -->` blocks. Prose outside those
markers is human-edited. Regenerate with
`make -C studies/04-gru-vs-disrnn-embedding-recovery`. Full contract in the
posthoc-reporting skill.

## Headline score

`likelihood_relative_to_groundtruth` = model eval NL ÷ generating-policy eval NL
(ceiling 1.0), logged by all three trainers. Recovery = R²/CCA of the learned
subject embedding against true per-subject parameters (Stages 1–2), and
classification accuracy of model variant/family from the embedding (Stages 3–4).
The `baseline_rl` runs are the correct-model-class **reference**, not a
competitor.
