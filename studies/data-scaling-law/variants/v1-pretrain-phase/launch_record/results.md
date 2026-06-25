# launch_record (v1-pretrain-phase main)

- W&B group: `mice-data-scaling-gru` (legacy; predates the `<variant>@<launch_id>` convention)
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker exp: `01KVPPMQ38NNT00870Q1QAT0XF` — 15 tasks on `ai1/octo-hub-onprem-h200`, priority `low`, preemptible, 1 GPU x 256GiB x 16 CPU
- Submitted: 2026-06-21 20:40 PT (2026-06-22T03:40:32 UTC)
- Settled: not queried (downstream analysis reads it as finished)
- Status: success
- Key numbers: 15 cells (D in {10, 30, 100, 300, 614} x 3 seeds, H=128); held-out per-trial likelihood 0.7219 -> 0.7268 as D 10 -> 614 (per FINAL_REPORT.md Result 1, v1 column)
- Feeds reports (indirectly, via `heldout-rerun-v1@*` offline re-evals — checkpoints from this group are fine-tuned on the held-out cohort): [r1](../../../analysis/reports/r1-heldout-scaling-curve.md), [r2](../../../analysis/reports/r2-per-mouse-repeated-measures.md), [r3](../../../analysis/reports/r3-bootstrap-cis.md), [r4](../../../analysis/reports/r4-zeroshot-vs-adapted.md), [r5](../../../analysis/reports/r5-fewshot-adaptation.md), [r8](../../../analysis/reports/r8-gru-vs-rl-baseline.md)
- Notes: dispatcher commit `1ef067be`; sweep file: `studies/data-scaling-law/sweep_data_scaling.yaml`. Repo docs (`FINAL_REPORT.md` Provenance, `README.md` variants table, `scaling_results.csv`, the `wandb_groups:` field in r8) refer to this launch as `v1-pretrain-phase@20260622-013415`. The submitted Beaker YAML, however, has `WANDB_RUN_GROUP=mice-data-scaling-gru`. Whether W&B exposes both names (alias / retag) or only one is not verified from local files; the YAML-baked value is what was actually shipped to the workers.
