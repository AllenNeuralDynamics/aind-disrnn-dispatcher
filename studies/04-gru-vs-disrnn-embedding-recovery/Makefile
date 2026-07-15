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
.PHONY: all r1 r2 stage1-prelim stage1-baseline stage2-recovery stage2-trajectory stage2b stage3-recovery stage4a-recovery stage4b-recovery

all: r1 stage1-prelim stage1-baseline stage2-recovery stage2-trajectory stage2b stage3-recovery stage4a-recovery stage4b-recovery

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

# Stage-2 combined recovery figure (3 square panels; likelihood + subject-param
# recovery + per-param bars), matching stage1_recovery_vs_baseline. Offline producer
# reads three committed curated inputs (likelihood comparison, GRU recovery, baseline
# recovery), each keyed by wandb_run_id. The baseline recovery CSV is produced OFF the
# Claude Science sandbox (compute_stage2_baseline_recovery.py, needs W&B/GCS) — it is
# committed here so `make` stays offline.
stage2b:
	$(PY) analysis/stage2b_flip_figure.py --flip-csv analysis/stage2b_likelihood_flip.csv --out analysis/figures/stage2b_likelihood_flip.png
	$(PY) analysis/stage2b_trajectory_figure.py --traj-csv analysis/stage2b_trajectory_recovery.csv \
		--emb-csv analysis/stage2b_embedding_trajectories.csv \
		--out analysis/figures/stage2b_session_trajectory.png

# NOTE: analysis/fetch_stage2b_embedding_trajectories.py is NOT part of `all` -- it is the
# one W&B-touching step in this figure's provenance (regenerates analysis/stage2b_embedding_trajectories.csv
# from the training code's own reconstruction). Re-run it manually only if the embedding
# example-subject coordinates need to be refreshed; `stage2b` itself stays fully offline.

stage2-recovery:
	$(PY) analysis/stage2_recovery_figure.py \
		--lik-csv analysis/stage2_likelihood_comparison.csv \
		--persession-csv analysis/stage2_persession_recovery.csv \
		--out analysis/figures/stage2_recovery_vs_baseline.png

stage2-trajectory:
	$(PY) analysis/stage2_trajectory_figure.py \
		--traj-csv analysis/stage2_trajectory_recovery.csv \
		--emb-csv analysis/stage2_embedding_trajectories.csv \
		--out analysis/figures/stage2_session_trajectory.png
# NOTE: analysis/fetch_stage2_embedding_trajectories.py is NOT part of `all` -- same
# one-off pattern as the stage2b embedding fetch: it is the only W&B-touching step in
# this figure's provenance. Re-run it manually only if the coordinates go stale.

# Stage-3 model-identity confusion (baseline vs GRU, real per-subject data) + fit-quality/
# per-session-recovery figures, matching the stage1/stage2/stage2b house convention.
# Offline producers, reading committed curated inputs (s3_baseline_modelselection.csv,
# stage3_gru_details.json, stage3_gru_recovery.csv, s3_baseline_likelihood.csv,
# stage3_baseline_likelihood_summary.csv, stage3_persession_recovery.csv). The baseline
# per-subject/per-session likelihood + recovery CSVs are produced OFF the Claude Science
# sandbox (compute_stage3_baseline_likelihood.py, compute_stage3_persession_recovery.py,
# fetch_stage3_gt_likelihood.py -- need the HPC filesystem + W&B) and committed here so
# `make` stays offline.
stage3-recovery:
	$(PY) analysis/stage3_confusion_figure.py \
		--ms-csv analysis/s3_baseline_modelselection_3true.csv \
		--gru-details-json analysis/stage3_gru_details.json \
		--baselines matched \
		--out analysis/figures/stage3_baseline_vs_gru_confusion.png
	$(PY) analysis/stage3_recovery_vs_baseline.py \
		--gru-csv analysis/stage3_gru_recovery.csv \
		--baseline-summary-csv analysis/stage3_baseline_likelihood_summary.csv \
		--persession-csv analysis/stage3_persession_recovery.csv \
		--out analysis/figures/stage3_recovery_vs_baseline.png
	$(PY) analysis/stage3_initial_param_scatter.py \
		--scatter-csv analysis/stage3_baseline_initial_param_scatter.csv \
		--out analysis/figures/stage3_baseline_initial_param_scatter.png
	$(PY) analysis/stage3_trajectory_figure.py \
		--trajectory-csv analysis/stage3_full_trajectory.csv \
		--baseline-fits-json analysis/stage3_baseline_fits.json \
		--gru-blind-csv analysis/stage3_gru_blind_predictions.csv \
		--gru-cond-csv analysis/stage3_gru_cond_predictions.csv \
		--out analysis/figures/stage3_trajectory_recovery.png

# stage4a-recovery regenerates analysis/figures/stage4a_recovery_combined.png from frozen
# curated inputs (embedding + GRU confusion details + GRU embed-size grid + baseline
# model-selection crosstab), keyed by immutable wandb_run_id inside the JSON/CSV -- offline,
# no W&B pull, so `make` stays offline. Also regenerates the per-session parameter
# recovery figure (baseline_rl vs GRU session-blind vs GRU session-conditioned) from
# stage4a_persession_recovery.csv, the Stage-4a analog of stage3_recovery_vs_baseline.png's
# panels b/c -- produced OFF-sandbox (compute_stage4a_persession_recovery.py needs the HPC
# filesystem + W&B) and committed here so `make` stays offline.
stage4a-recovery:
	$(PY) analysis/stage4a_recovery_figure.py \
		--emb-csv analysis/stage4a_embedding.csv \
		--gru-details-json analysis/stage4a_gru_details.json \
		--gru-recovery-csv analysis/stage4a_gru_recovery.csv \
		--ms-csv analysis/stage4a_baseline_modelselection.csv \
		--out analysis/figures/stage4a_recovery_combined.png
	$(PY) analysis/stage4a_persession_figure.py \
		--persession-csv analysis/stage4a_persession_recovery.csv \
		--out analysis/figures/stage4a_persession_recovery.png

# stage4b-recovery regenerates analysis/figures/stage4b_recovery.png (per-session
# family-switching: subject embedding recovers stable Dirichlet mixture weights; session
# conditioning adds ~nothing for per-session family decoding). Offline producer reads two
# frozen committed inputs -- stage4b_recovery_grid.csv (per-run summary) and
# stage4b_recovery.json (in-container recovery output: per-family mix-weight R2 +
# per-session confusion matrix), keyed by wandb_run_id (sweep nptb5bam) -- no W&B pull, so
# `make` stays offline. The JSON is produced OFF-sandbox (needs the wrapper loader + W&B)
# and committed here as the frozen authoritative copy.
stage4b-recovery:
	$(PY) analysis/stage4b_recovery_figure.py \
		--grid-csv analysis/stage4b_recovery_grid.csv \
		--recovery-json analysis/stage4b_recovery.json \
		--out analysis/figures/stage4b_recovery.png
