"""Single source of truth for the W&B summary keys this study reads.

Contract between the wrapper's per-run held-out analysis
(``aind-disrnn-wrapper/code/post_training_analysis/``, the producer) and this
study's cross-run aggregation (``timing_scaling.py``, the consumer). The wrapper
writes these keys into each run's W&B summary; a schema change here is a one-file
diff.

The wrapper commit that defines these keys is pinned in the study's
``environment.lock`` and stamped into every analysis JSON as
``_meta.wrapper_git_sha``.

This study's output is UNCHANGED from study-01 (2-way choice, ``n_classes=2``),
so the held-out likelihood is directly comparable across all five arms and to
study-01. The only change per arm is which previous-trial inputs are appended.
"""
from __future__ import annotations

# --- primary metric: held-out-MOUSE choice likelihood -----------------------
# The value from the final ``auto_heldout_finetune`` pass (fine-tune the subject
# embedding on a held-out mouse's early sessions, then predict its other
# sessions). This is the study's ONLY headline metric; within-subject LL below is
# a diagnostic, never a headline (Han, 2026-08-22).
HELDOUT = "heldout/final/eval_likelihood"
# Older runs (pre-fix) logged the un-suffixed key; read as a fallback only.
HELDOUT_FALLBACK = "heldout/eval_likelihood"

# --- within-subject diagnostic: eval on held-out SESSIONS of TRAINED mice ----
# This is the metric best_eval selects the checkpoint on. It saturates with D
# and, below D~100, collapses to the point where it no longer tracks held-out
# LL (see r2) -- so it is reported only as a selection-reliability diagnostic.
WITHIN = "checkpoint/eval_likelihood"
WITHIN_TRAIN = "checkpoint/train_likelihood"        # for the train-eval overfit gap
SELECTED_STEP = "checkpoint/step"                    # the best_eval-selected checkpoint
FINAL_STEP = "_step"                                 # last training step reached

# --- config fields used to classify a run into (arm, D, seed) ---------------
# arm is derived from data.timing_features: enabled=False -> OFF; shuffle=True ->
# SHUF; else ON if both reaction_time & lick_counts, RT if only reaction_time,
# LICK if only lick_counts. D = len(resolved_subject_ids).
CFG_TIMING = "data/timing_features"                  # nested: enabled/shuffle/reaction_time/lick_counts
CFG_SUBJECTS = "resolved_subject_ids"                # list; D = len(...)

# Convenience grouping.
ALL_HELDOUT = [HELDOUT, HELDOUT_FALLBACK]
