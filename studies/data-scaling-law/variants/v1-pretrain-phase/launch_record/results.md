# launch_record (v1-pretrain-phase main)

- W&B group: `mice-data-scaling-gru` (legacy; predates the `<variant>@<launch_id>` convention)
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker exp: `01KVQ7EJ3C5YJ8FJVNJB8C8N36` — 15 tasks on `ai1/octo-hub-onprem-h200`, mixed-priority: 11 preemptible (autoResume) + 4 allocated/non-preemptible (the 4 largest-D runs); `WANDB_MODE=offline`, synced to W&B post-hoc (see `notes.md` for the `-r1` run-name detail)
- Submitted: 2026-06-22 01:34 PT (2026-06-22T08:34:15 UTC, decoded from the ULID)
- Settled: not queried (downstream analysis reads it as finished; `notes.md` says 15/15 in W&B)
- Status: success
- Key numbers: 15 cells (D in {10, 30, 100, 300, 614} x 3 seeds, H=128); held-out per-trial likelihood 0.7219 -> 0.7268 as D 10 -> 614 (per FINAL_REPORT.md Result 1, v1 column)
- Feeds reports (indirectly, via `heldout-rerun-v1@*` offline re-evals — checkpoints from this group are fine-tuned on the held-out cohort): [r1](../../../analysis/reports/r1-heldout-scaling-curve.md), [r2](../../../analysis/reports/r2-per-mouse-repeated-measures.md), [r3](../../../analysis/reports/r3-bootstrap-cis.md), [r4](../../../analysis/reports/r4-zeroshot-vs-adapted.md), [r5](../../../analysis/reports/r5-fewshot-adaptation.md), [r8](../../../analysis/reports/r8-gru-vs-rl-baseline.md)
- Notes: this dir's `beaker_resumable.json` + `experiment_resumable_submitted.yaml` record an **earlier cancelled** attempt (`01KVPPMQ38NNT00870Q1QAT0XF`, 2026-06-21 20:40 PT, dispatcher `1ef067be`) — the launch_record auto-save was not re-run for the six subsequent relaunches that culminated in `01KVQ7EJ3C5YJ8FJVNJB8C8N36` ~5 h later. Wrapper for the completed experiment: `bdb326d` (per `notes.md`). Sweep file: `studies/data-scaling-law/sweep_data_scaling.yaml`. Full relaunch backstory in `studies/data-scaling-law/README.md` status log (2026-06-22 entries).
