#!/bin/bash
# Smallest real cohort, minimal sampler: proves the production path end to end, not science.
#SBATCH --job-name=hb-smoke
#SBATCH --partition=aind
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-hb-gpu/bin
cd /home/han.hou/code/aind-disrnn-wrapper/code
nvidia-smi --query-gpu=name --format=csv,noheader
DISRNN_META_STUDY=08-hb-vs-gru-heldout \
DISRNN_META_VARIANT=one-stage-ladder \
DISRNN_META_NOTE="path smoke test through run_hpc" \
$B/python -m run_hpc \
  data=mice_snapshot_scaling \
  model=hb_hattori \
  data.subject_ratio=0.016 \
  model.num_warmup=30 model.num_samples=30 model.num_chains=1 \
  'model.few_shot_k=[0]' \
  seed=0
