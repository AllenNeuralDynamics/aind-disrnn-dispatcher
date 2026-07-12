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
# analysis/recovery_summary.{json,csv} + the three figures, and regenerates the
# BEGIN/END blocks in both reports. So r1 and r2 share one producer run.
#
# WANDB_API_KEY is only needed to RE-PULL the grids from the W&B project (on an
# authenticated HPC node, where wandb.Api works); the committed grids are the
# default source of truth, so `make` runs offline.

PY ?= python
.PHONY: all r1 r2

all: r1

# r1 and r2 are produced together by recovery_report.py (one figure set, both
# report blocks). Listed separately for discoverability; both run the same command.
r1 r2:
	$(PY) analysis/recovery_report.py
