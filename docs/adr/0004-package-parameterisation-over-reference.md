# Parameter conventions follow aind-dynamic-foraging-models, not the reference Stan model

Where the reference Stan implementation and `aind-dynamic-foraging-models` disagree on how a
parameter is expressed, the package wins. Concretely: `forget_rate_unchosen` (decay, `0` means
no forgetting) rather than Stan's `aF` (retention, `1` means no forgetting, and
`aF = 1 - forget_rate_unchosen`); and `biasL` with the package's sign rather than Stan's
`bias = -biasL`.

The reason is comparability. The whole point of the hierarchical Bayesian baseline is to sit on
one axis with the existing per-session MLE fits produced by that same package. Any translation
layer between the two parameterisations is a place where an inversion or a sign flip silently
corrupts a cross-model comparison, and both traps are present here.

## Consequences

Our reported parameters will not be numerically identical to the published ones even for the
same fitted model: `forget_rate_unchosen` is `1 - aF`, and `biasL` has the opposite sign from
the paper's `bias`. Anyone reconciling our numbers against the paper needs the mapping table in
`docs/design-hb-baseline.md` §2.
