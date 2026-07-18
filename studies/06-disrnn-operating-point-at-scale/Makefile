# studies/06-disrnn-operating-point-at-scale/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# Reports regenerate OFFLINE from the committed grid (analysis/grid.csv); `pull` re-pulls it
# from W&B (needs WANDB_API_KEY). The scaling producer overlays THIS study's tuned curve against
# study 05's fixed-penalty curve and study 01's GRU curve, so the reconciliation is one figure.
#
# NOTE: producers (analysis/scaling_report.py, analysis/pull_grid.py) are added when the
# tuned-fixed-point grid finishes and analysis/grid.csv is committed. Until then these targets
# are placeholders — see analysis/reports/INDEX.md for status.

PY ?= python
.PHONY: all r1 pull

all: r1

# r1 = tuned-point scaling curve vs 05 (fixed penalty) vs 01 (GRU). Reads committed grid.csv.
r1:
	$(PY) analysis/scaling_report.py

# Refresh analysis/grid.csv from W&B, filtered to this study's group (needs WANDB_API_KEY).
# Not part of `all` — the committed CSV is the source of truth so reports regenerate offline.
pull:
	$(PY) analysis/pull_grid.py
