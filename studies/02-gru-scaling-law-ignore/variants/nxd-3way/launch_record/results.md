# launch_record — nxd-3way

- **W&B group:** original launch label `nxd-3way@20260703-094210` — but that first
  launch was **superseded** by later relaunches (Beaker/HPC hedging + the held-out
  re-score) and contributed **no** runs to the settled grid. The 13 groups that
  actually fed the final numbers are stamped in `analysis/scaling.json` →
  `_meta.wandb_groups` (and mirrored in the r1/r2 report frontmatter).
- **W&B project:** https://wandb.ai/AIND-disRNN/mice_ignore_scaling
- **Beaker exp:** `01KWMEF9TMM391Q5BC15KZXDVQ` (resumable 48-task grid; superseded
  `01KWMDT0ETCM6QW6XADAKEQXP8`). Cluster split: H16/64/128 → g6e/l40s, H256 →
  octo-hub-onprem-h200. Later cells hedged/backfilled on H200; the exact re-score
  ran on H200 (`01KX7Q4X...`, `01KX7T0T...`, validation `01KX7H7V...`).
- **Submitted:** 2026-07-03 09:42 PT
- **Settled:** 2026-07-11 PT (after the held-out re-score correction)
- **Status:** ✅ success — 48/48 cells (16 H×D cells × 3 seeds), all carrying
  correct held-out LR-engaged + ignore-class metrics.
- **Key numbers:** best cell H=256/D=614 held-out L/R-engaged **0.73149**
  (SEM 0.00002), still climbing with D; H=16 flattens (0.7164→0.7210). Ignore
  detection PR-AUC ~0.61→0.64 with D; recall capped ~0.47. Full grid in
  `analysis/scaling.json`.
- **Feeds reports:** r1 (LR-engaged scaling), r2 (ignore detection scaling).
- **Notes.** This grid took a long path (Beaker→HPC migration, held-out output-root
  and seed bugs, an OOM cascade at D=614/H256, and a launcher YAML-fold bug — all
  fixed). The one correctness issue that reached the numbers: 12 cells were
  ignore-backfilled via the `DISRNN_RESTORE_FROM_RUN_ID` restore path, which
  **resumed base training** and over-trained the model, giving wrong held-out
  metrics. Those 12 cells were re-scored exactly (held-out finetune-and-eval only,
  no base training) via `resume_heldout_beaker.py`; the fix reproduces the native
  LR-engaged value to <1e-6. Full audit + the 12 cell ids in
  `../../../analysis/provenance/backfill_history.md` and `rescore_launched.json`.
  The sweep specs for every wave (HPC split, Beaker hedges, backfills, image
  test) are under `_sweeps/`.
