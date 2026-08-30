#!/bin/bash
#SBATCH --job-name=hb-smoke
#SBATCH --partition=aind
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -e
B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-hb-gpu/bin
D=/home/han.hou/code/aind-disrnn-dispatcher/studies/08-hb-vs-gru-heldout
cd "$D"
nvidia-smi --query-gpu=name --format=csv,noheader
# Smallest real cohort, minimal sampler: proves the production path, not the science.
$B/python -u run_hb.py \
  --estimator one_stage --subject-ratio 0.016 --seed 0 \
  --num-warmup 30 --num-samples 30 --num-chains 1 --few-shot-k 0 \
  --launch-id smoke-$(TZ=America/Los_Angeles date +%Y%m%d-%H%M%S) \
  --note "path smoke test: real loader + batched held-out scoring + W&B keys" \
  --output "$D/smoke_d10_wired.json"
