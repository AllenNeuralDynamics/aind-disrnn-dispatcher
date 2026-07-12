# embedding-recovery — future directions

## Data-generation performance (partially addressed)

Synthetic data is generated on-the-fly per job. At mice scale (200-300 subjects ×
40 sessions × 650 trials ≈ 5-8M trials) generation is non-trivial because each
(subject, session) is an independent pure-Python forager rollout.

**DONE — multiprocessing.** `HierarchicalCognitiveAgents` now fans subject
simulation across a `spawn` process pool (`generation_workers`, default auto =
min(SLURM_CPUS_PER_TASK, num_subjects); env `DISRNN_GEN_WORKERS`). Measured on the
Allen `aind` CPU partition: 200-subject generation dropped from >13 min (serial,
never finished before a 13-min cancel) to **~4.7 min with 8 workers**. Output is
byte-identical to serial (determinism test `test_end_to_end_serial_equals_parallel`).

**TODO — frozen snapshot (generate-once, load-many).** The multiprocessing fix
speeds up *each* job but every job still regenerates its dataset. Across the Stage-1
grid there are only **4 distinct datasets** (one per `num_subjects` ∈
{50,100,200,300}), yet the 36 GRU + 4 baseline_rl jobs regenerate them ~10× each.
A frozen snapshot would eliminate this redundancy:

- Generate each `num_subjects` dataset once (deterministic, so the snapshot ≡
  on-the-fly), persist the merged `DatasetRNN` tensors + ground-truth CSV, and add
  a loader mode that loads the snapshot instead of regenerating.
- **Tension (already flagged):** `ai1/octo-hub-gcp-h100` cannot reach AWS S3, which
  is why we generate on-the-fly in the first place. A snapshot therefore can't live
  only on S3 if we want to keep using the idle H100s. Options:
  1. **Beaker dataset mount** — upload each snapshot as a Beaker dataset and mount
     it into the gcp-h100 job (no S3 dependency at runtime). Cleanest for GPU.
  2. **Allen filesystem** (`/allen/...` or scratch) for the HPC route — snapshots
     load directly; does not help gcp-h100.
  3. Keep on-the-fly as the portable default and treat the snapshot as an opt-in
     accelerator for large repeated sweeps.
- Because generation is deterministic, the snapshot is a pure caching optimization
  — it changes nothing scientifically, only wall time. Worth doing before the full
  36-point (and disRNN) grids if compute pressure warrants; not required for
  correctness.

## Grid completion

Stage-1 first pass runs a LEAN CPU grid (hidden{16,64} × embed{2,4}) because
GRU-on-CPU is slow (~3-4h/job at hidden=64). The dropped cells (hidden=256,
embed=8) should be added on GPU (gcp-h100) once the node disk-full clears — GPU
makes each job minutes, so the full 36-point grid + disRNN replication become cheap.
