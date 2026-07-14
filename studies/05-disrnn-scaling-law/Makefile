# studies/05-disrnn-scaling-law/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# scaling_report.py is the single producer for BOTH reports (r1 = held-out scaling,
# r2 = sparsity + multiplier): it reads the committed grid analysis/grid.csv, writes
# analysis/summary.{json,csv} + the three figures, and regenerates the BEGIN/END blocks.
# So r1 and r2 share one producer run.
#
# WANDB_API_KEY is only needed to RE-PULL the grid from W&B (analysis/pull_grid.py); the
# committed CSV is the default source of truth, so `make` runs offline.

PY ?= python
.PHONY: all r1 r2 r4 pull

all: r1 r4

# r1 and r2 are produced together by scaling_report.py (one figure set, both report blocks).
# Listed separately for discoverability; both run the same command.
r1 r2:
	$(PY) analysis/scaling_report.py

# r4 = generative behavioral match. Unlike r1/r2 this one reads W&B directly (the rollout
# scalars are per-run summaries, not a committed grid), so it needs WANDB_API_KEY.
r4:
	$(PY) analysis/generative_match.py

# Refresh analysis/grid.csv from W&B (needs WANDB_API_KEY). Not part of `all` — the
# committed CSV is the source of truth so reports regenerate offline.
pull:
	$(PY) analysis/pull_grid.py
