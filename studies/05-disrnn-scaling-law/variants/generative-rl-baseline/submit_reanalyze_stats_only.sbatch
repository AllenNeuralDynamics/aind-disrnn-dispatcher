#!/bin/bash
# Recompute switch + history-dependent STATS ONLY (no figures, no re-simulation) from cached
# session histories. See reanalyze_stats_only.py docstring: two prior attempts timed out because
# they generated figures r4 never reads (per-session scatter plots, ~20-25 min each); the stats
# themselves compute in well under a minute once the cached histories are loaded.
#
#   sbatch --export=ALIAS=ctt,DIR=/path/out/ctt,VARIANT=$PWD \
#          --output=/path/out/reanalyze-stats-ctt-%j.log submit_reanalyze_stats_only.sbatch
#
#SBATCH --partition=aind
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=rl-reanalyze-stats

set -euo pipefail

: "${ALIAS:?set ALIAS=ctt|bari|hattori}"
: "${DIR:?set DIR=<the run_rl_rollout.py --out/ALIAS dir, on a SHARED fs>}"
: "${VARIANT:?set VARIANT=<this variant dir>}"

source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu

python "$VARIANT/reanalyze_stats_only.py" --alias "$ALIAS" --dir "$DIR"
