"""Single source of truth for the W&B summary keys this study reads.

These keys are the *contract* between the wrapper's per-run analysis
(``aind-disrnn-wrapper/code/post_training_analysis/``, which writes them into each run's W&B
summary) and this study's cross-run aggregation. The wrapper is the producer; the scripts here are
the consumer. Keeping the vocabulary in one place means a wrapper schema change is a one-file diff,
and a renamed key becomes a loud error rather than a silently dropped run.

Mirrors ``studies/01-gru-scaling-law/analysis/wandb_keys.py`` so the disRNN's generative numbers are
read with the identical key set the GRU's were -- that is what makes r4's model-vs-model comparison
legitimate.
"""
from __future__ import annotations


# --- generative: switch-triggered curve -----------------------------------
def switch_corr(stat: str) -> str:
    return f"combined/switch_triggered/quantitative_summary/subject_mean/{stat}/correlation"


def switch_mse(stat: str) -> str:
    return (f"combined/switch_triggered/delta_significance_summary/{stat}/"
            "subject_balanced_error_summary/mean_squared_error")


# --- generative: history-dependent p(switch) ------------------------------
def hist_corr(pattern: str, n_back: int) -> str:
    return f"combined/history_dependent/quantitative_summary/subject_mean/{pattern}/{n_back}/correlation"


def hist_mse(pattern: str, n_back: int) -> str:
    return (f"combined/history_dependent/delta_significance_summary/{pattern}/{n_back}/"
            "subject_balanced_error_summary/mean_squared_error")
