# studies/06-disrnn-operating-point-at-scale/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# r1 regenerates OFFLINE from committed study 03 + 05 data (analysis/data/d100_selection.csv +
# ../05.../analysis/grid.csv) -- no WANDB_API_KEY needed, zero new compute.
# r2 regenerates OFFLINE from the committed mult-d-grid (analysis/grid.csv); `pull` re-pulls it
# from W&B (needs WANDB_API_KEY) -- run before r2 to get the freshest numbers while the grid is
# still in flight. r2 overlays this study's live grid against study 05's fixed-penalty curve and
# study 01's GRU curve.

PY ?= python
.PHONY: all r1 r2 pull

all: r1 r2

# r1 = penalty selection (Kevin's in-sample-vs-held-out plot), zero new compute.
r1:
	$(PY) analysis/model_selection.py

# r2 = mult-d-grid scaling surface, live while the grid runs. `make pull` first for fresh data.
r2:
	$(PY) analysis/scaling_report.py

# Refresh analysis/grid.csv from W&B, filtered to this study's group (needs WANDB_API_KEY).
# Not part of `all` — the committed CSV is the source of truth so reports regenerate offline.
pull:
	$(PY) analysis/pull_grid.py
