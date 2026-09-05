---
id: r2
slug: author-aligned-baselines
status: live
authors: [han, codex]
wandb_groups:
  - grossman-meta-learning@20260905-124420
  - chen-rlck@20260905-123624
  - zid-history-kernel@20260905-123624
inputs:
  script: analysis/report_author_baselines.py
  data:
    - analysis/author_baseline_results.json
    - analysis/matched_half_results.json
  figure: analysis/fig_author_baseline_likelihood.png
reproduce: make -C studies/09-gru-cross-species-transfer r2
---

# Result 2 — Author-aligned behavioral baselines

This result asks whether the transferred GRU advantage survives comparison with the behavioral
model selected in each source paper, rather than only our common sticky Q-learning control.

<!-- BEGIN result-2 -->
Pending successful result freeze.
<!-- END result-2 -->

## Interpretation

These are refits on our matched split, not copied likelihoods from the papers. That keeps test
trials, information budget, scoring, and subject-level fitting identical across model families.
