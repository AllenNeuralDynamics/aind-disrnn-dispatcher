"""W&B summary-key contract for 04-gru-vs-disrnn-embedding-recovery (comment-only).

The producer (analysis/recovery_report.py) runs OFFLINE from the committed curated
grids; this file documents which W&B summary keys those grids were derived from, so a
re-pull (on HPC, where wandb.Api works) reads the right fields. Project: embedding_recovery.

Per-run summary keys read:
  likelihood_relative_to_groundtruth   # headline fit score (model NL / ground-truth NL)
  avg_eval_likelihood_groundtruth      # generating-policy eval NL (denominator)
  _step, _runtime                      # progress / ETA only

Config (nested) read for cell identity:
  model.architecture.session_encoding_type      # {none, scalar}
  model.architecture.subject_embedding_size      # {4, 8, 16}
  data.value.{task, agent, num_subjects, num_sessions_per_subject, num_trials,
              seed, heldout_session_mode, heldout_frac}   # GT regeneration

Recovery metrics are NOT logged to W&B; they are computed post-hoc by the wrapper's
analysis/stage4b_recovery.py (model-agnostic: extract_subject_embeddings_from_params +
compute_session_conditioned_context_dataframe) and curated into the *_grid.csv files here.

Output artifact naming: <mtype>-output-<run_id>  (gru-output / disrnn-output).
"""
