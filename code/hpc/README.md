# Allen HPC Launching

This directory owns the AI1 Allen HPC launch path for disRNN. The dispatcher
repo is the control plane for Code Ocean, Beaker, and Allen HPC; the wrapper
repo remains the training/runtime payload.

Expected local layout:

```bash
/path/to/parent/
  aind-disrnn-dispatcher/
  aind-disrnn-wrapper/
```

## Setup

Create the runtime environments from the wrapper repo:

```bash
cd /path/to/parent/aind-disrnn-wrapper

conda create -n disrnn-cpu python=3.12 -y
conda activate disrnn-cpu
pip install -e .

conda create -n disrnn-gpu python=3.12 -y
conda activate disrnn-gpu
pip install -e ".[gpu]"
```

The launcher is always invoked from the CPU environment. The SLURM script then
activates `disrnn-cpu` or `disrnn-gpu` on the compute node based on
`--mode cpu|gpu`.

Create the per-user SLURM env file in the dispatcher repo:

```bash
cd /path/to/parent/aind-disrnn-dispatcher
cp code/hpc/slurm/user.env.example code/hpc/slurm/user.env
# edit code/hpc/slurm/user.env
```

Optionally source it from your shell startup file so manual `sbatch` commands
pick up the same settings:

```bash
echo 'source /path/to/aind-disrnn-dispatcher/code/hpc/slurm/user.env' >> ~/.bashrc
```

`SBATCH_*` variables are read by `sbatch`. `CONDA_SH` is used by the SLURM
scripts to activate the runtime environment.

## W&B Sweep Launch

Run from the dispatcher repo root:

```bash
conda activate disrnn-cpu

python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/scaling_disrnn.yaml \
  --mode cpu

python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/scaling_disrnn.yaml \
  --mode gpu
```

Useful launch variants:

```bash
# Inspect planned commands without creating a sweep or submitting SLURM jobs.
python code/launch_hpc.py --mode gpu --dry-run

# Bounded validation: one array task, one W&B agent run.
python code/launch_hpc.py --mode gpu --sbatch-extra=--array=0-0 --agent-count 1

# Request a specific GPU tier.
python code/launch_hpc.py --mode gpu --gpu-type a100

# Use a non-sibling wrapper checkout.
python code/launch_hpc.py --mode gpu --wrapper-root /path/to/aind-disrnn-wrapper
```

`code/launch_hpc.py` creates the W&B sweep, injects dispatcher and wrapper git
lineage into every run as Hydra `+meta.*` overrides, computes `AGENT_COUNT` for
grid sweeps, submits the SLURM array, and submits a small cleanup job that marks
the W&B sweep finished after the array drains.

## Manual SLURM Use

Manual use skips lineage injection and `AGENT_COUNT` auto-computation:

```bash
wandb sweep code/hpc/sweeps/scaling_disrnn.yaml
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-disrnn-wrapper \
  code/hpc/slurm/wandb_sweep_gpu.slurm <SWEEP_ID>
```

Hydra multirun scripts are also available when you want deterministic config
enumeration without W&B sweep orchestration:

```bash
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-disrnn-wrapper \
  code/hpc/slurm/hydra_multirun_cpu.slurm

sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-disrnn-wrapper \
  code/hpc/slurm/hydra_multirun_gpu.slurm
```

## Monitoring

```bash
squeue -u $USER
scontrol show job <jobid>
sacct -j <array_jobid> --format=JobID,State,ExitCode,Start,End
tail -f $HOME/logfile/job_<jobid>.out
scancel <jobid>
```

If you launched with `--no-autostop`, stop the W&B sweep manually after the
agents finish:

```bash
wandb sweep --stop <SWEEP_ID>
```

## GPU Tier

The GPU SLURM scripts default to `--gres=gpu:v100:1` and `--array=0-5`, matching
the six-GPU per-user cap on the `aind` QOS. Override the tier per launch with
`--gpu-type` for W&B sweeps or `sbatch --gres=gpu:<type>:1` for manual Hydra
multiruns.
