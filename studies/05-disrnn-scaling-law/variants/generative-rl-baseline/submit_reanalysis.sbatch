#!/bin/bash
# Recompute switch + history stats from ALREADY-SIMULATED histories (no re-simulation).
# The first pass (submit.sbatch) hit its 6h wall clock during history significance testing;
# the switch curve and both session histories were already cached to disk. This just re-derives
# the stats from those caches with the subject-level bootstrap disabled (it is the ~4h+ cost and
# is decorative for r4's point estimates -- see reanalyze_from_cache.py docstring).
#
#   sbatch --export=ALIAS=ctt,DIR=/path/out/ctt,VARIANT=$PWD \
#          --output=/path/out/reanalyze-ctt-%j.log submit_reanalysis.sbatch
#
#SBATCH --partition=aind
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=rl-reanalyze

set -euo pipefail

: "${ALIAS:?set ALIAS=ctt|bari|hattori}"
: "${DIR:?set DIR=<the run_rl_rollout.py --out/ALIAS dir, on a SHARED fs>}"
: "${VARIANT:?set VARIANT=<this variant dir>}"

source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu

# The load-bearing line: disables the subject-level bootstrap (250 resamples/pattern by default),
# which is what made the history-dependent pass ~10x more expensive than the switch curve. Point
# estimates (correlation, RMSE) do not depend on it -- only the CIs the bootstrap would add.
export DISRNN_SUBJECT_BOOTSTRAP_RESAMPLES=0

python "$VARIANT/reanalyze_from_cache.py" --alias "$ALIAS" --dir "$DIR"
