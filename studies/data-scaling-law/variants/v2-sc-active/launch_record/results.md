# launch_record (v2-sc-active main)

- W&B group: `v2-sc-active@20260622-144622`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker exp: `01KVRMSAAJTRSJMFV5JT7JAP6X` — 15 tasks, **split clusters**:
  - tasks 000–003 on `ai1/octo-hub-aws-l40s`: priority `normal`, non-preemptible, 1 GPU x 90GiB x 12 CPU (fits one L40s bundle)
  - tasks 004–014 on `ai1/octo-hub-onprem-h200`: priority `low`, preemptible, 1 GPU x 256GiB x 16 CPU
- Submitted: 2026-06-22 14:28 PT (2026-06-22T21:28:50 UTC)
- Settled: not queried (downstream analysis reads it as finished)
- Status: success — supersedes 5 prior submission attempts (`01KVRGFG`, `01KVRHZ5`, `01KVRKGA`, `01KVRKRT`, `01KVRKVN`; see `beaker_resumable.json` note)
- Key numbers: 15 cells (D in {10, 30, 100, 300, 614} x 3 seeds, H=128); held-out per-trial likelihood 0.7218 -> 0.7282 as D 10 -> 614; v2-v1 large-D Delta ~ +0.00146 (per FINAL_REPORT Result 1).
- Feeds reports:
  - **directly** (`nxd_scaling.py` reads this group): [r7](../../../analysis/reports/r7-nxd-joint-scaling-grid.md) (H=128 row, D in {10, 30, 100, 614})
  - **indirectly** (via `heldout-rerun-v2@*` and `heldout-rerun-v2-mature2@*` offline re-evals): [r1](../../../analysis/reports/r1-heldout-scaling-curve.md), [r2](../../../analysis/reports/r2-per-mouse-repeated-measures.md), [r3](../../../analysis/reports/r3-bootstrap-cis.md), [r4](../../../analysis/reports/r4-zeroshot-vs-adapted.md), [r5](../../../analysis/reports/r5-fewshot-adaptation.md), [r6](../../../analysis/reports/r6-sc-stage-mature-only.md), [r8](../../../analysis/reports/r8-gru-vs-rl-baseline.md)
- Notes: dispatcher commit `a1b92b50`; sweep: `studies/data-scaling-law/variants/v2-sc-active/sweep.yaml`. The L40s split exists because the 256GiB request on L40s forced a 3-GPU bundle assignment — dropping mem to 90GiB on L40s fits the single-GPU bundle (per AGENTS §10 sizing).
