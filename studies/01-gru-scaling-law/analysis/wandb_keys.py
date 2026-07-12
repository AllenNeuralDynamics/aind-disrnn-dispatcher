"""Single source of truth for the W&B summary keys this study reads.

These keys are the *contract* between the wrapper's per-run analysis
(``aind-disrnn-wrapper/code/post_training_analysis/``, which writes them into
each run's W&B summary) and this study's cross-run aggregation. The wrapper is
the producer; the scripts here are the consumer. Keeping the vocabulary in one
place means a wrapper schema change is a one-file diff to review, and the
`require` helper turns a renamed/missing key into a loud error instead of a
silently dropped run. See docs/repo-split-plan.md "Analysis contract" for why
this lives here rather than in a shared cross-repo schema package.

Wrapper commit that defines these keys is pinned in the study's
``environment.lock`` and stamped into every analysis JSON as
``_meta.wrapper_git_sha``.
"""
from __future__ import annotations

# --- generative: switch-triggered curve -----------------------------------
def switch_corr(stat: str) -> str:
    return f"combined/switch_triggered/quantitative_summary/subject_mean/{stat}/correlation"


def switch_mse(stat: str) -> str:
    return (f"combined/switch_triggered/delta_significance_summary/{stat}/"
            "subject_balanced_error_summary/mean_squared_error")


SWITCH_CORR_OVERALL = "combined/switch_triggered/quantitative_summary/subject_mean/overall/correlation"

# --- generative: history-dependent p(switch) ------------------------------
def hist_corr(pattern: str, n_back: int) -> str:
    return f"combined/history_dependent/quantitative_summary/subject_mean/{pattern}/{n_back}/correlation"


def hist_mse(pattern: str, n_back: int) -> str:
    return (f"combined/history_dependent/delta_significance_summary/{pattern}/{n_back}/"
            "subject_balanced_error_summary/mean_squared_error")


# --- held-out likelihood (scaling curve) ----------------------------------
# Primary first, then legacy fallback; a run legitimately carries one or the other.
HELDOUT_LL_KEYS = ("heldout/final/eval_likelihood", "heldout/eval_likelihood")


_MISSING = object()


def require(summary, key: str, run_label: str):
    """Return ``summary[key]``, raising loudly if the key is absent.

    Use for keys every relevant run *must* have; a miss means the wrapper
    schema changed (see this module). For keys that some runs legitimately
    lack, use ``summary.get`` and skip, but guard against *all* runs missing
    (that is a rename, not a partial run).
    """
    val = summary.get(key, _MISSING)
    if val is _MISSING or val is None:
        raise KeyError(
            f"{run_label}: expected W&B summary key {key!r} is missing "
            f"(wrapper schema changed? keys are defined in analysis/wandb_keys.py)"
        )
    return val
