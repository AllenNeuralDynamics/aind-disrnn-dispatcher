# The population level pools both location and scale

Each subject has a location `mu_p` and a session-spread `sigma`. The population level sits over
**both**: `mu_p_m ~ N(M, S)` and `log sigma_m ~ N(m_ls, s_ls)`.

The obvious-looking alternative is to pool only `mu_p` and leave each subject's `sigma_m` under
a fixed prior, which is what you get by naively stacking a level on top of the published
two-level model. Rejected because `sigma_m` sets how far a fresh session may deviate from its
subject's mean, and therefore directly controls the predictive spread when scoring a held-out
session. Leaving it unpooled means a held-out subject's `sigma_m` is informed only by a fixed
prior — degrading precisely the zero-shot and low-`k` few-shot cells the baseline exists to
measure.

## Consequences

If `sigma_m` turns out poorly identified in the recovery study, the fallback is a single
`sigma` shared across all subjects. That is a stronger assumption — every subject equally
consistent day to day — and discards a potentially real individual difference, so it should be
adopted only on evidence.
