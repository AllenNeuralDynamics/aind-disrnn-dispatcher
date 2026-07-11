# launch_record (nxd-grid main)

- W&B group: `nxd-grid@20260623-102649`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker exp: `01KVTRAF21E73WDWEX8MDRYM9E` — 27 tasks on `ai1/octo-hub-onprem-h200`, priority `low`, preemptible, 1 GPU x 256GiB x 16 CPU
- Submitted: 2026-06-23 10:26 PT (2026-06-23T17:26:53 UTC)
- Settled: not queried (downstream `nxd_scaling.py` reads it as finished)
- Status: success
- Key numbers: 27 cells (H in {16, 64, 256} x D in {10, 100, 614} x 3 seeds). The H=128 row of the joint r7 grid is reused from `v2-sc-active`; the D=30 column for these N comes from `launch_record_d30_g6e` (sibling dir).
- Feeds reports: [r7](../../../analysis/reports/r7-nxd-joint-scaling-grid.md) (non-H128 rows x D in {10, 100, 614})
- Notes: dispatcher commit `99aa2fca`; sweep: `studies/data-scaling-law/variants/nxd-grid/sweep.yaml`. Same SC-active recipe (lambda forward, gated early-stop @70k) as v2-sc-active so the H=128 column can be reused for the joint grid.
