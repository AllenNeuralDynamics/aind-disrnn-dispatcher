# Study 08 — HB vs GRU on held-out subjects

Fits the hierarchical Bayesian cognitive baseline (`HB-Hattori2019`) on **study 01's
cohort**, so its held-out likelihood is directly comparable with the GRU numbers already in
`../01-gru-scaling-law/scaling_results.csv`.

Recovery and validation live in `aind-dynamic-foraging-models` (standalone, synthetic data).
This study is the real-data comparison only.

## The stack

Which repo owns which part of the fit, and where the held-out boundary sits, is
documented once for the whole stack rather than per study:
**[docs/diagrams/hb-stack](../../docs/diagrams/)** — open
[the interactive version](https://raw.githack.com/AllenNeuralDynamics/aind-disrnn-dispatcher/main/docs/diagrams/hb-stack.html)
to pan, zoom, search and trace a single relationship.

The figures in this folder are this study's own results; the stack diagram is
architecture and is true whether or not this study exists.

## Cohort

`data.subject_ratio` against the ~614 pool, matching study 01's D dial:

| ratio | D | GRU held-out likelihood (seeds 0/1/2) |
|---|---|---|
| 0.049 | ~30 | 0.7248 / 0.7252 / 0.7250 |
| 0.163 | ~100 | 0.7264 / 0.7264 / 0.7260 |

Same snapshot (`20260603`), curricula, `min_sessions=10`, `heldout_every_n=5`,
`mature_only=false`, `ignore_policy=exclude`. The GRU seed spread of ~0.0004 is the natural
yardstick for whether a difference means anything.

## Variants index

| variant | what differs | status | W&B group |
|---|---|---|---|
| [`one-stage-ladder`](variants/one-stage-ladder/notes.md) | `subject_ratio` 0.016/0.049/0.163 → D≈10/30/101, one-stage estimator | running (2026-08-29) | `hb-one_stage@20260829-181251` |

**Deviation from study-conventions:** this study logs into study 01's W&B project
`mice_data_scaling` rather than its own. Deliberate — the results must land on study 01's
scaling curve and be read by its `analyze_scaling.py`; a separate project would turn the
comparison into a manual join. Recorded in the variant's notes.

## Running

HB goes through the **same entrypoint as every other model** — there is no HB-specific
script, and therefore nothing HB-specific to keep in sync:

    python -m run_hpc data=mice_snapshot_scaling model=hb_hattori data.subject_ratio=0.049

`run_hpc` is shared by HPC/SLURM and Beaker, so the same command works on both backends.
Data selection comes from `code/config/data/mice_snapshot_scaling.yaml` — the file study 01
uses — rather than being restated, so the two cannot drift apart.

The ladder rungs:

    sbatch variants/one-stage-ladder/production.sbatch 0.016 d10  "$LAUNCH_ID"
    sbatch variants/one-stage-ladder/production.sbatch 0.049 d30  "$LAUNCH_ID"
    sbatch variants/one-stage-ladder/production.sbatch 0.163 d100 "$LAUNCH_ID"

Both estimators exist; `estimator=one_stage` is the default and the reference. Two-stage
matches it to within 0.00027 on held-out likelihood while costing ~5x the compute, so it is
a fallback for scales where the joint fit will not converge. Override with
`model.estimator=two_stage`.

## Status

D~30 launched 2026-08-29. D~100 follows if D~30 is clean.
