# Three-level hierarchy: population, subject, session

The published model is two-level — subject-level hyperparameters governing session-level
parameters — and is fit independently per subject (`fit_animal(animalID)`, with `N` = that
subject's sessions). It therefore has no cohort level and nothing transfers to an unseen subject.
We add a population level above the subject so that held-out subjects can be scored the way the
GRU/disRNN models are, which is the entire purpose of building this baseline.

Because the subject-level location parameters are already unconstrained (bounded values are
obtained by a normal-CDF transform), the population level is a plain Gaussian over them:
`mu_p_m ~ N(M, S)`.

## Consequences

Setting `M = 0, S = 1` recovers the published per-subject model exactly, since `Phi(X)` with
`X ~ N(0,1)` is `Uniform(0,1)` — the paper's stated uniform priors on bounded subject-level
parameters. This makes the published model a nested special case, and gives a clean
regression test.

Anyone comparing our parameter estimates against the paper's must know the extra level exists:
per-subject estimates are shrunk toward the cohort, so they will not match the published
per-subject fits exactly, by design.
