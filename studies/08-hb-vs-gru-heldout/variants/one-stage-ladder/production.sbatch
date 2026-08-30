#!/bin/bash
# HB scaling ladder, launched exactly like any other model in this stack.
#
#   sbatch production.sbatch <subject_ratio> <label> <launch_id> [note]
#
# The same command runs on Beaker: run_hpc is the shared entrypoint for both backends
# (see aind-disrnn-wrapper/beaker/smoke.yaml), so there is no HB-specific plumbing.
#SBATCH --job-name=hb-prod
#SBATCH --partition=aind
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "usage: production.sbatch <subject_ratio> <label> <launch_id> [note]" >&2
  exit 2
fi
RATIO=$1
LABEL=$2
LAUNCH_ID=$3
NOTE=${4:-"HB one-stage scaling ladder rung ${LABEL}; matched-conditioning held-out likelihood against study 01's GRU at the same D"}

B=/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-hb-gpu/bin
cd /home/han.hou/code/aind-disrnn-wrapper/code
nvidia-smi --query-gpu=name --format=csv,noheader

DISRNN_META_STUDY=08-hb-vs-gru-heldout \
DISRNN_META_VARIANT=one-stage-ladder \
DISRNN_META_LAUNCH_ID="$LAUNCH_ID" \
DISRNN_META_LABEL="$LABEL" \
DISRNN_META_NOTE="$NOTE" \
$B/python -m run_hpc \
  data=mice_snapshot_scaling \
  model=hb_hattori \
  data.subject_ratio="$RATIO" \
  seed=0
