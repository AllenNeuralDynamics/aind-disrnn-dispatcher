# Held-out likelihood is pointwise lppd, averaged in probability space

Held-out score for the hierarchical Bayesian models is the log pointwise predictive density:
for each trial, average the observed choice's probability across posterior draws, then take
the log.

    sum_t log[ (1/L) * sum_l p_l(y_t | y_<t) ]

Averaging in log space instead — `(1/L) sum_l log p_l` — is a different and smaller quantity by
Jensen's inequality, and systematically understates the model. Scoring at the posterior mean or
MAP overstates it. The correct value lies between, and this is the standard `lppd` underlying
WAIC and PSIS-LOO.

## Considered options

The strictly correct posterior predictive for a session marginalises the whole session at once,
`log[(1/L) sum_l prod_t p_l(y_t | y_<t)]`, since the session-level parameters are shared across
that session's trials rather than redrawn per trial. Rejected: across ~650 trials the per-draw
session likelihoods span hundreds of nats, so the log-sum-exp collapses onto the single best
draw and the estimate is unusably high-variance at any feasible `L`. The pointwise form is also
the one commensurable with the neural models' existing per-trial metric.

## Consequences

The collapse from `L` draws to one number happens *before* the log, and the result drops
into the wrapper's existing `_compute_normalized_likelihood` unchanged, so hierarchical
Bayesian, GRU, disRNN, and baseline-RL all land on one axis.

This is easy to "fix" incorrectly. A future contributor who changes the averaging to log space,
or who plugs in a point estimate, will silently change every reported number.
