---
name: hpc-launch
description: Launch and monitor disRNN training on the Allen on-premise SLURM HPC (AI1) — launch_hpc.py W&B sweep arrays, manual sbatch/Hydra multirun, dry-run/smoke-test patterns, GPU tiers, extend/restore and held-out re-scoring, squeue/sacct monitoring. Use whenever running jobs on the Allen cluster via srun/sbatch rather than Beaker.
---

# Launching on Allen on-prem HPC (SLURM)

Canonical detail: `code/hpc/README.md`. **If this skill and the README conflict, the
README wins.**

## Hard rules first

1. **Never run computation on the login node** (where you run). Everything heavy goes
   through `sbatch`/`srun` — including smoke tests.
2. Invoke the launcher from the `disrnn-cpu` conda env
   (`/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-cpu`). The SLURM script
   activates `disrnn-cpu` or `disrnn-gpu` on the compute node per `--mode`.
3. Two-repo layout: the wrapper repo is expected as a sibling
   (`../aind-disrnn-wrapper`); override with `--wrapper-root`.
4. One-time setup per user: `cp code/hpc/slurm/user.env.example code/hpc/slurm/user.env`
   and edit (`SBATCH_*` vars for sbatch, `CONDA_SH` for env activation).

## Check available resources FIRST (mandatory for large jobs)

**Before launching any large job (> 4 GPUs / > 4 concurrent array tasks), check
schedulable capacity** and route to whichever backend (HPC vs Beaker) has room.
Hard rule (AGENTS.md §10).

```bash
python code/check_gpu_availability.py --hpc     # this partition (default: aind)
python code/check_gpu_availability.py           # HPC + Beaker, to load-balance
```

`sinfo` free counts include `drain`/`down`/reserved nodes. The script reports the
real figure — `CfgTRES.gres/gpu − AllocTRES.gres/gpu` on non-drained nodes —
broken down by GPU type (a100/h200/v100/l40s/…). HPC often has genuinely idle GPUs
when all Beaker hub clusters are saturated/cordoned, so it is the natural overflow
for large GPU grids. Requires the Allen network (login node, or the sandbox with
VPN up).

## Standard launch (W&B sweep + SLURM array)

From the dispatcher repo root:

```bash
conda activate disrnn-cpu
python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/<sweep>.yaml \
  --mode gpu \
  --label <short-label> \
  --note "why this run exists + what we want to learn"
```

`launch_hpc.py` creates the W&B sweep, injects dispatcher+wrapper git lineage as Hydra
`+meta.*` overrides, computes `AGENT_COUNT` for grid sweeps, submits the SLURM array,
and submits a cleanup job that stops the sweep after the array drains
(`--no-autostop` to keep it open for more agents).

Useful variants:

```bash
# Inspect planned commands; nothing is created or submitted.
python code/launch_hpc.py --mode gpu --dry-run

# End-to-end smoke test: one tiny run on one CPU array task.
python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/synthetic_disrnn_smoke.yaml \
  --mode cpu --sbatch-extra=--array=0-0 --agent-count 1

# Bounded validation: one grid point of the default sweep.
python code/launch_hpc.py --mode cpu --sbatch-extra=--array=0-0 --agent-count 1

# Specific GPU tier (default gres is gpu:v100:1, array 0-5 — the 6-GPU aind-QOS cap).
python code/launch_hpc.py --mode gpu --gpu-type a100
```

## Manual SLURM routes (skip lineage injection + AGENT_COUNT)

```bash
wandb sweep code/hpc/sweeps/<sweep>.yaml
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-disrnn-wrapper \
  code/hpc/slurm/wandb_sweep_gpu.slurm <SWEEP_ID>

# Deterministic Hydra multirun without a W&B sweep controller:
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-disrnn-wrapper \
  code/hpc/slurm/hydra_multirun_gpu.slurm   # or _cpu.slurm
```

## Resuming, extending & re-scoring runs

HPC `aind` jobs are **not preempted** (queue / fair-share scheduling, not Beaker's
preemptible-priority tiers), so the "resume after preemption" scenario does not arise
here — that mechanism is Beaker-only. The other two work the same as on Beaker (full
detail: wrapper `../aind-disrnn-wrapper/code/TRAINING.md` §1.5):

- **Extend a finished run to a longer horizon.** Set
  `model.training.restore_from_run_id=<source W&B run name>` (or env
  `DISRNN_RESTORE_FROM_RUN_ID`) and a **larger** `n_steps`. `run_hpc.py` downloads the
  source run's `<mtype>-output-<run_id>` artifact (`mtype` ∈ {`disrnn`,`gru`}) into
  `outputs/` before training, so it resumes from the checkpoint and skips warmup.
  Trainer-agnostic. Prereq: the source run must have FINISHED (its `training-output`
  artifact is COMMITTED).
- **Re-score a finished run's held-out stage only — no re-training.** Runs the held-out
  fine-tune off the existing checkpoint and re-injects `heldout/*` into the original
  W&B run — used to backfill metrics added *after* a run trained. The committed
  reference implementations both live under the wrapper's `code/`: the Beaker port
  `code/resume_heldout_beaker.py` (`--run-id <wandb_run_id>`, inside a container reaching
  GCS + W&B) and the HPC original `code/resume_heldout.py` (`--model-dir <dir> --wandb-run-id
  <id>`, run on a compute node that can read the checkpoint tree + reach W&B).

## Monitoring

```bash
squeue -u $USER
scontrol show job <jobid>
sacct -j <array_jobid> --format=JobID,State,ExitCode,Start,End
tail -f $HOME/logfile/job_<jobid>.out
scancel <jobid>
wandb sweep --stop <SWEEP_ID>   # only needed after --no-autostop
```

When reporting results, link the W&B sweep/run URL and stamp times in Seattle time
(`TZ=America/Los_Angeles date`) — AGENTS.md §7.
