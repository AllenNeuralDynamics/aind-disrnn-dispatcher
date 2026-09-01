# h128-dscan-onprem — the on-prem half of the timing-ON D-scan

Experiment `01M0M997B9XNCY0W1Z9FK69GA8` (9 tasks originally; 12 job records after
preemption/autoResume), W&B group `h128-dscan-onprem`, project
`mice_rt_lick_scaling`.

This is **not a separate arm** — it is the same timing-ON recipe as `h128-dscan`,
split onto `ai1/octo-hub-onprem-h200` because the aws-h200 queue could not take
all 15 cells at once. Same WRAPPER_REF, same sweep knobs, same W&B project, so
the curve reassembles across the two groups.

## Recorded deviation: the split correlates cluster with D

The cells were split **by D** (aws kept the cells already training; onprem took
D≈100/300/614), which means cluster and the swept axis are correlated — a
cluster-level artifact would alias onto the curve's shape. Mitigation:
`subject_ratio=0.163` was run on **both** clusters as a direct cross-check.

This mistake informed the later arms: `h128-dscan-off-bigD`/`-off-smallD`,
`-shuffled`, `-rt-only` and `-licks-only` all list **both** S3-capable H200
clusters in every task, so placement is orthogonal to D. Split by *tier*, never
by the swept axis.

## Cluster fact established here

`ai1/octo-hub-onprem-h200` **does** reach the AWS S3 parquet cache — a probe read
the snapshot session table and printed `S3_PARQUET_OK sessions=23868`. Its earlier
reputation for failing was a stale-image problem, not a connectivity one. See
`code/beaker/README.md` and the `beaker-launch` skill, both corrected.

## Held-out metric

These runs trained before the held-out timing fix, so they carried
`heldout=MISSING` and were recovered by the standalone re-score
(`resume_heldout_beaker.py`) rather than retrained.
