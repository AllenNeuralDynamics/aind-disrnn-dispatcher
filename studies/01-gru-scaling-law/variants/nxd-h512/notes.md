# Variant nxd-h512

**What differs from the siblings.** Adds a new capacity row `hidden_size=512` to
the N×D joint-scaling grid (r7), across the same four D columns
(`subject_ratio ∈ {0.016, 0.049, 0.163, 1.0}` → D ≈ 10 / 30 / 100 / 614) × 3
seeds = **12 tasks**. Every other knob is identical to `nxd-grid` /
`v2-sc-active` (scalar session conditioning, λ-forward full SC @50k,
`n_steps=150k`, `lr=1e-5`, gated early-stop @70k, length bucketing,
`checkpoint_run_heldout_eval=false`).

**Why.** r7 shows D saturates by ~100 mice and the N×D interaction is weak up to
H=256. This tests whether that story holds one capacity octave higher (H=256→512).
Unlike the H=128 row (reused from `v2-sc-active`), there is **no existing H=512
column** — all 12 cells are trained fresh here.

**Expected cost (from measured nxd-grid wall-clock).** Existing runs early-stop at
~90.5k steps; per-step time rises with H (H16→256: 0.264→0.350 s/step). Fitting
that to H=512 gives ~0.36 s/step (log-linear) to ~0.59 s/step (quadratic, GRU
gate matmuls are O(H²)) → **per-job median ≈ 10–15 h**, D-dependent:
~11 h (D=10) → ~15 h (D=100) → ~18–20 h (D=614, worst cell). With the ~6-GPU
`aind` QOS concurrency, 12 jobs run in ~2 waves ⇒ **~1.5 days** of run time plus
queue. NOTE: H=512 is larger than anything trained so far (max H=256); confirm the
per-task memory bundle is sufficient at launch.

**Launch (HPC SLURM; Beaker was saturated at launch time).**
```
conda activate disrnn-cpu
python code/launch_hpc.py \
  --sweep-yaml studies/01-gru-scaling-law/variants/nxd-h512/sweep.yaml \
  --mode gpu \
  --sbatch-extra='--array=0-11 --time=24:00:00' \
  --label nxd-h512 \
  --note "H=512 capacity row for the r7 N×D grid: does D-saturation / weak N×D interaction persist one octave above H=256?"
```
`--array=0-11` ⇒ 12 array tasks; launcher computes AGENT_COUNT=ceil(12/12)=1
(one W&B run per task). `--time=24:00:00` overrides the slurm script's 5 h
default (H=512 needs 10–20 h). Add `--gpu-type <tier>` to pick a faster GPU than
the v100 default if capacity allows.

**Status.** 🟡 RESUBMITTED on h200 (2026-07-20 19:53 PT). First launch FAILED.

- **Launch 1 (a100) — FAILED, all 12 OOM.** sweep `5iy5qnb2`, group
  `nxd-h512@20260718-192434`, array `23241259`. Queued ~19 h, scheduled
  2026-07-19 14:50 PT, every task crashed at **step 0** with JAX
  `RESOURCE_EXHAUSTED` (GPU OOM). Root cause: the `aind` a100s are **40 GB PCIE**
  cards; H=512 at `batch_size=2048` exceeds 40 GB at peak (the log shows a ~14.6 GiB
  allocation failing on top of a ~14 GiB working set, with XLA pre-reserving most of
  the card on top of that). It fits on h200 (141 GB). (An earlier note called these
  "MIG ~16–20 GB slices" — that was an unverified guess and is wrong; the cards are
  full 40 GB, H=512 simply needs >40 GB.) SLURM mislabeled 11/12 COMPLETED because
  `wandb agent` exits 0 on child crash; task 10 flagged OUT_OF_MEMORY.
  W&B confirms `_step=0`/None on all 12 → no training, nothing to salvage.
- **Launch 2 (h200) — SUBMITTED, queued (PENDING).** Same sweep.yaml, `--gpu-type h200`, batch size
  UNCHANGED (2048) — h200 full 141 GB fits H=512, keeping optimization identical
  to the rest of the r7 grid. sweep `ajsw1a8h`
  (https://wandb.ai/AIND-disRNN/mice_data_scaling/sweeps/ajsw1a8h),
  group `nxd-h512@20260720-195322`, array `23263174` (`--array=0-11`,
  `--gres=gpu:h200:1 --time=24:00:00`), autostop `23263175`.
  `sbatch --test-only` est. start 2026-07-23 00:28 PT (~2.2 d; h200 was the
  fastest of h200/l40s/a100 as well as the only full-mem fit).

- **Launch 2 outcome (2026-07-23):** 9/12 cells FINISHED cleanly on h200
  (D=10/30/100 × 3 seeds, early-stopped 80k–90k steps, 8–13 h each across 3 waves
  of 4 — the per-user h200 QOS cap is 4, verified via `sacctmgr`). The 3 **D=614**
  seeds (tasks 9/10/11, `subject_ratio=1.0` = full 614-mouse dataset) FAILED with a
  **host-RAM OOM** (SLURM `OUT_OF_MEMORY`, exit `0:125`, `slurmstepd oom_kill`;
  MaxRSS 81–95 GB, true peak > the `--mem=128G` cgroup limit) during data loading at
  step 0 — NOT a GPU OOM (h200 was fine).
- **Launch 3 (d614 resubmit) — SUBMITTED, queued.** Only the 3 D=614 seeds, on h200,
  `--mem=256G` (batch size unchanged), via `sweep_d614.yaml`. sweep `aw04872p`
  (https://wandb.ai/AIND-disRNN/mice_data_scaling/sweeps/aw04872p), group
  `nxd-h512@20260723-125804`, array `23302804` (`--array=0-2 --time=24:00:00
  --mem=256G`), autostop `23302805`.

**W&B groups.** The H=512 row now spans TWO groups (nxd_scaling.py must collect both):
`nxd-h512@20260720-195322` (D=10/30/100, 9 cells) + `nxd-h512@20260723-125804`
(D=614, 3 cells), project `AIND-disRNN/mice_data_scaling`.

**Downstream analysis wiring (pending; do AFTER runs finish).** To add the H=512
row to r7, `analysis/nxd_scaling.py` needs: (1) `TARGET_HS` → `[16,64,128,256,512]`;
(2) add the group `nxd-h512@20260718-192434` to `NXD_GROUPS`. The D=300 note is
unaffected (H=512 uses the same four rectangular D columns). Then re-run the r7
producer chain and update the report + README Variants index.
