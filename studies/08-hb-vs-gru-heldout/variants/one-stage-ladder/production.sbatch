#!/bin/bash
#SBATCH --job-name=hb-prod
#SBATCH --partition=aind
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
# Fail fast rather than silently writing one_stage_.json or landing in a stray W&B group.
if [ $# -lt 3 ]; then
  echo "usage: production.sbatch <subject_ratio> <label> <launch_id> [note]" >&2
  exit 2
fi
RATIO=$1
LABEL=$2
LAUNCH_ID=$3
NOTE=${4:-"HB one-stage scaling ladder rung ${LABEL}; matched-conditioning held-out likelihood against study 01's GRU at the same D"}
B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-hb-gpu/bin
D=/home/han.hou/code/aind-disrnn-dispatcher/studies/08-hb-vs-gru-heldout
cd "$D"
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$D/artifacts/$LABEL"
$B/python -u run_hb.py \
  --estimator one_stage --subject-ratio "$RATIO" --seed 0 --few-shot-k 0 \
  --launch-id "$LAUNCH_ID" \
  --note "$NOTE" \
  --artifact-dir "$D/artifacts/$LABEL" \
  --output "$D/one_stage_${LABEL}.json"
