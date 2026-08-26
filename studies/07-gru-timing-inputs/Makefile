# studies/07-gru-timing-inputs/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per
# docs/posthoc-analysis.md "Makefile convention". Invoke from this directory
# (studies/07-gru-timing-inputs/); the producer is cwd-independent via
# HERE = Path(__file__).parent.
#
# timing_scaling.py is the single producer for BOTH reports (r1 = block
# decomposition, r2 = selection reliability): it pulls the live W&B project
# mice_rt_lick_scaling, writes analysis/timing_scaling.json (+ .csv + the three
# study-root figures), and regenerates the BEGIN/END blocks in both reports via
# update_reports.py. So r1 and r2 share one producer run.
#
# Requires WANDB_API_KEY (or ~/.netrc wandb creds) to reach the W&B project.

PY ?= python
.PHONY: all r1 r2

all: r1

# r1 and r2 are produced together by timing_scaling.py (one W&B pull, three
# figures, both report blocks). Listing them separately is for discoverability;
# both run the same command.
r1 r2:
	$(PY) analysis/timing_scaling.py
