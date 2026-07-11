"""Single source of truth for the W&B summary keys this study reads.

Contract between the wrapper's per-run held-out analysis
(``aind-disrnn-wrapper/code/post_training_analysis/``, the producer) and this
study's cross-run aggregation (``scaling.py``, the consumer). The wrapper writes
these keys into each run's W&B summary; a schema change here is a one-file diff.

The wrapper commit that defines these keys is pinned in the study's
``environment.lock`` and stamped into every analysis JSON as
``_meta.wrapper_git_sha``.

All keys live under the ``heldout/final/`` prefix — the values from the final
``auto_heldout_finetune`` pass (fine-tune the subject embedding on a held-out
mouse's sessions, then predict its other sessions). This study's 3-way output
(L/R/ignore) means the held-out likelihood is decomposed into an engaged-trial
L/R component and an ignore-class-detection component (see the metric caveat in
the study README): the raw 3-way NL is NOT comparable to the 2-way L/R NL.
"""
from __future__ import annotations

# --- primary scaling metric: conditional L/R likelihood on ENGAGED trials ---
# This is the like-for-like comparator to the 2-way data-scaling-law curve:
# the model's L/R prediction scored only on trials the mouse engaged with
# (animal_response != ignore), so the chance baseline matches the 2-way 1/2.
LR_ENGAGED = "heldout/final/eval_likelihood_LR_engaged"

# --- engagement (binary engage-vs-ignore) likelihood ------------------------
ENGAGE = "heldout/final/eval_likelihood_engage"

# --- ignore-class detection metrics (rare positive class ~5-10% base rate) --
# PR-AUC is the headline detection metric (robust to the class imbalance);
# recall/precision/f1 characterise the operating point; base_rate is the
# no-skill floor for context.
IGNORE_PR_AUC = "heldout/final/engage_ignore_pr_auc"
IGNORE_RECALL = "heldout/final/engage_ignore_recall"
IGNORE_PRECISION = "heldout/final/engage_ignore_precision"
IGNORE_F1 = "heldout/final/engage_ignore_f1"
IGNORE_BASE_RATE = "heldout/final/engage_ignore_base_rate"

# Raw 3-way normalised likelihood (chance 1/3) — recorded but NOT plotted as a
# scaling axis because it is not comparable to the 2-way curve. Use LR_ENGAGED
# for the choice-prediction comparison and the ignore-* keys for detection.
RAW_3WAY = "heldout/eval_likelihood"

# Convenience groupings used by scaling.py.
ALL_HELDOUT_FINAL = [
    LR_ENGAGED, ENGAGE,
    IGNORE_PR_AUC, IGNORE_RECALL, IGNORE_PRECISION, IGNORE_F1, IGNORE_BASE_RATE,
]
