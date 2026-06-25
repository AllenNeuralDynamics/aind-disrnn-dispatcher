# launch_record (nxd-grid D=30 gap-fill, AWS g6e)

- W&B group: `nxd-grid@20260624-141106`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker exp: `01KVXQHTSH0780FFBCJAGAXA2E` — 9 tasks on `ai1/octo.ai-aws-g6e` (AWS L40S), priority `low`, preemptible, 1 GPU x 90GiB x 12 CPU. Per AGENTS §10, `octo.ai-aws-g6e` is the **only** non-hub cluster we ship low-priority preemptible jobs to.
- Submitted: 2026-06-24 14:11 PT (2026-06-24T21:11:09 UTC)
- Settled: between 2026-06-24 23:18 PT and ~23:30 PT (watcher log tail at 23:18 PT still shows 9/9 `running`; the analysis re-run at ~23:30 PT picked up all D=30 cells with no nans)
- Status: success — filled all 9 D=30 cells for H in {16, 64, 256} (H=128/D=30 already in `v2-sc-active`)
- Key numbers: 9 cells (H in {16, 64, 256} x D=30 x 3 seeds). N=256/D=30 cell completed at 0.7251 (was `nan` in the r7 grid before this launch). Refitted r7: alpha 1.23 -> 1.30, beta 0.58 -> 0.56, E ~ 0.729; log-log interaction p=0.311.
- Feeds reports: [r7](../../../analysis/reports/r7-nxd-joint-scaling-grid.md) (the D=30 column for non-H128 rows)
- Notes: dispatcher commit `a1857c02`; sweep: `studies/data-scaling-law/variants/nxd-grid/sweep_d30.yaml`. Watcher (`analysis/watch_nxd_d30.py`) chained `nxd_scaling.py` -> `rl_baseline.py` -> `update_final_report_nxd.py` once cells settled; the watcher process was killed after the first successful regen.
