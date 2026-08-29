# Study 08 — HB vs GRU on held-out subjects

Fits the hierarchical Bayesian cognitive baseline (`HB-Hattori2019`) on **study 01's
cohort**, so its held-out likelihood is directly comparable with the GRU numbers already in
`../01-gru-scaling-law/scaling_results.csv`.

Recovery and validation live in `aind-dynamic-foraging-models` (standalone, synthetic data).
This study is the real-data comparison only.

## Cohort

`data.subject_ratio` against the ~614 pool, matching study 01's D dial:

| ratio | D | GRU held-out likelihood (seeds 0/1/2) |
|---|---|---|
| 0.049 | ~30 | 0.7248 / 0.7252 / 0.7250 |
| 0.163 | ~100 | 0.7264 / 0.7264 / 0.7260 |

Same snapshot (`20260603`), curricula, `min_sessions=10`, `heldout_every_n=5`,
`mature_only=false`, `ignore_policy=exclude`. The GRU seed spread of ~0.0004 is the natural
yardstick for whether a difference means anything.

## Running

    python run_hb.py --estimator two_stage --subject-ratio 0.049 --output two_stage_d30.json
    python run_hb.py --estimator one_stage --subject-ratio 0.049 --output one_stage_d30.json

Both estimators are run because two-stage is an approximation to one-stage; it is promoted
to full scale only if it matches. See `docs/design-hb-baseline.md` decision 15.

## Status

D~30 launched 2026-08-29. D~100 follows if D~30 is clean.
