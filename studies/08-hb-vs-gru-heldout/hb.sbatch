#!/bin/bash
#SBATCH --job-name=hb-d30
#SBATCH --partition=aind
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -e
ESTIMATOR=${1:-two_stage}
RATIO=${2:-0.049}
LABEL=$(python3 -c "print(f'd{round(614*$RATIO):d}')")
OUT=${3:-/home/han.hou/code/aind-disrnn-dispatcher/studies/08-hb-vs-gru-heldout/${ESTIMATOR}_${LABEL}.json}
B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-hb-gpu/bin
cd /home/han.hou/code/aind-disrnn-dispatcher/studies/08-hb-vs-gru-heldout
nvidia-smi --query-gpu=name --format=csv,noheader
$B/python -u run_hb.py --estimator "$ESTIMATOR" --subject-ratio "$RATIO" --output "$OUT"
