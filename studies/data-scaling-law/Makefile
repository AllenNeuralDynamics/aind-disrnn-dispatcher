# studies/data-scaling-law/Makefile
#
# One PHONY target per report; `all` regenerates everything that has a
# committed producer. Per docs/posthoc-analysis.md "Makefile convention".
#
# Invoke from this directory (studies/data-scaling-law/). All producers
# are cwd-independent via HERE = Path(__file__).parent.
#
# Cross-producer figures (known violation of "one producer per artifact",
# documented in docs/posthoc-analysis.md):
#   - fig_nxd_scaling.png  : nxd_scaling.py writes bare; rl_baseline.py overlays
#   - fig_scaling_v1_v2.png: build_report.py writes bare; rl_baseline.py overlays
#   - fig_fewshot_curve.png, fig_zeroshot_vs_d.png: rl_baseline.py overlays
# Targets that produce these figures chain rl_baseline.py last so the
# committed PNGs end up with the overlay.

.PHONY: all r1 r2 r3 r4 r5 r6 r7 r8

# r2 is subsumed by r1 (same producer); r6 has no committed producer.
all: r1 r3 r4 r5 r7 r8

# r1 + r2 share build_report.py (writes report_data.json, fig_scaling_v1_v2.png
# [bare], fig_per_subject_delta.png). rl_baseline.py restores the RL overlay
# on fig_scaling_v1_v2.png.
r1 r2:
	python analysis/build_report.py
	python analysis/rl_baseline.py

r3:
	python analysis/bootstrap_scaling.py

# r4 and r5 data JSONs are ad-hoc (no committed producer); only the figures'
# RL overlay refreshes via rl_baseline.py.
r4 r5:
	python analysis/rl_baseline.py

# r6 producer not wired; mature_sc_verdict.json/.md are ad-hoc.
r6:
	@echo "r6: producer not wired; mature_sc_verdict.json/.md are ad-hoc"

# r7 chains: nxd_scaling.py (data + bare figure) -> rl_baseline.py (overlay) ->
# update_final_report_nxd.py (rewrites the BEGIN/END block in the report).
r7:
	python analysis/nxd_scaling.py
	python analysis/rl_baseline.py
	python analysis/update_final_report_nxd.py

# r8: rl_baseline.py produces rl_baseline.json + fig_rl_paired.png;
# update_r8.py rewrites the paired-table block in the report.
r8:
	python analysis/rl_baseline.py
	python analysis/update_r8.py
