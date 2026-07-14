#!/bin/bash
# Generative rollout of one per-mouse RL baseline (r4's reference lines).
#
# CPU-only (RL simulation, no GPU) -> HPC SLURM, not Beaker (AGENTS §13 load-balancing).
# One job per baseline; each simulates all 18,124 sessions of all 614 mice from that mouse's own
# fitted parameters, through the wrapper's own baseline_rl_analysis engine.
#
#   sbatch --export=ALIAS=ctt,OUT=/path/out submit.sbatch
#   sbatch --export=ALIAS=bari,OUT=/path/out submit.sbatch
#   sbatch --export=ALIAS=hattori,OUT=/path/out submit.sbatch
#
#SBATCH --partition=aind
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=rl-generative
#SBATCH --output=%x-%j.log

set -euo pipefail

: "${ALIAS:?set ALIAS=ctt|bari|hattori}"
: "${OUT:?set OUT=<output dir>}"

source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu
export WANDB_API_KEY="$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/run_rl_rollout.py" --alias "$ALIAS" --out "$OUT/$ALIAS"
