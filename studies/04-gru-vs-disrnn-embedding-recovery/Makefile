# studies/04-gru-vs-disrnn-embedding-recovery/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per the
# posthoc-reporting convention. Invoke from this directory; the producer is
# cwd-independent via HERE = Path(__file__).parent.
#
# recovery_report.py is the SINGLE producer for BOTH reports (r1 = GRU ladder,
# r2 = disRNN replication): it reads the committed curated grids
# (analysis/ladder_results.csv, analysis/stage4b_recovery_grid.csv,
# analysis/disrnn_stage4a_recovery_grid.csv), writes
# analysis/recovery_summary.{json,csv} + two figures (fig_ladder, fig_disrnn_stage4a),
# and regenerates the BEGIN/END blocks in both reports. So r1 and r2 share one run.
#
# stage1_recovery_figure.py is a SECOND offline producer: it reads the committed
# stage1_recovery_grid.csv + stage1_recovery_scatter_n200_e4.csv and writes
# analysis/figures/stage1_recovery_preliminary.png (embedded in r1). Offline, no W&B.
#
# WANDB_API_KEY is only needed to RE-PULL the grids from the W&B project (on an
# authenticated HPC node, where wandb.Api works); the committed grids are the
# default source of truth, so `make` runs offline.

PY ?= python
.PHONY: all r1 r2 stage1-prelim stage1-baseline

all: r1 stage1-prelim stage1-baseline

# r1 and r2 are produced together by recovery_report.py (one figure set, both
# report blocks). Listed separately for discoverability; both run the same command.
r1 r2:
	$(PY) analysis/recovery_report.py

# Stage-1 embedding-size recovery figure (preliminary; embedded in r1). Separate
# producer + curated CSVs, keyed by immutable artifact version_id per the
# posthoc-reporting 'freeze the numbers' rule.
stage1-prelim:
	$(PY) analysis/stage1_recovery_figure.py

# Stage-1 GRU-vs-correct-model-baseline recovery figure (3 square panels). Offline
# producer reads three committed curated inputs (GRU + baseline recovery scores,
# gt-likelihood), each keyed by wandb_run_id per the 'freeze the numbers' rule.
stage1-baseline:
	$(PY) analysis/make_recovery_figures.py \
		--gru-csv analysis/stage1_gru_recovery.csv \
		--baseline-csv analysis/stage1_baseline_recovery.csv \
		--gt-json analysis/stage1_gt_likelihood.json \
		--out analysis/figures/stage1_recovery_vs_baseline.png
