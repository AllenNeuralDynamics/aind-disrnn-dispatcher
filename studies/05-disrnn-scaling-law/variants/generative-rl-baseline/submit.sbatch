#!/bin/bash
# Generative rollout of one per-mouse RL baseline (r4's reference lines).
#
# CPU-only (RL simulation, no GPU) -> HPC SLURM, not Beaker (AGENTS §13 load-balancing).
# One job per baseline; each simulates all 18,124 sessions of all 614 mice from that mouse's own
# fitted parameters, through the wrapper's own baseline_rl_analysis engine.
#
# OUT must be on a SHARED filesystem (/allen/...), never node-local scratch (/local1/...):
# the batch job lands on a different node than the one you submitted from.
#
#   sbatch --export=ALIAS=ctt,OUT=/allen/.../rlgen,VARIANT=$PWD \
#          --output=/allen/.../rlgen/rl-ctt-%j.log submit.sbatch
#
#SBATCH --partition=aind
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=rl-generative

set -euo pipefail

: "${ALIAS:?set ALIAS=ctt|bari|hattori}"
: "${OUT:?set OUT=<output dir on a SHARED fs>}"
# sbatch copies this script to a spool path, so BASH_SOURCE cannot locate run_rl_rollout.py.
: "${VARIANT:?set VARIANT=<this variant dir, i.e. the dir holding run_rl_rollout.py>}"

source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu
export WANDB_API_KEY="$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")"

python "$VARIANT/run_rl_rollout.py" --alias "$ALIAS" --out "$OUT/$ALIAS"
