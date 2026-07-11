# studies/beta-scan/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per
# docs/posthoc-analysis.md "Makefile convention". Invoke from this directory
# (studies/beta-scan/); producers are cwd-independent via HERE = Path(__file__).parent.
#
# beta_scan_report.py is the single producer for BOTH reports (r1 = bottleneck
# sparsity, r2 = held-out transfer): it reads the committed clean grid
# analysis/beta_scan_final_grid.csv, writes analysis/beta_scan_summary.{json,csv}
# + the two figures, and regenerates the BEGIN/END blocks in both reports via
# update_reports.py. So r1 and r2 share one producer run.
#
# WANDB_API_KEY is only needed to RE-PULL the grid CSV from the W&B project
# (see analysis/backfill_sparsity_metrics.py); the committed CSV is the default
# source of truth, so `make` runs offline.

PY ?= python
.PHONY: all r1 r2

all: r1

# r1 and r2 are produced together by beta_scan_report.py (one figure set, both
# report blocks). Listed separately for discoverability; both run the same command.
r1 r2:
	$(PY) analysis/beta_scan_report.py
