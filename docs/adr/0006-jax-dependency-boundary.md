# NumPyro is an optional extra, and the JAX code does not import the numpy foragers

`aind-dynamic-foraging-models` is pure numpy/scipy at `requires-python = ">=3.9"`, with a
3.9/3.10 CI matrix. JAX requires `>=3.10`. Rather than bump the floor and break the existing
matrix, NumPyro goes behind a `bayes` optional extra beside the existing `dev` and `rl`, and a
separate CI job installs `[bayes]` on 3.11+ and runs only the hierarchical Bayesian tests.

Additionally, the JAX likelihood **must not import the numpy foragers at runtime** — only in
tests.

## Consequences

The runtime-import rule is the non-obvious half, and it exists to protect the correctness
anchor. Having given up the published Stan implementation as an oracle (ADR-0001), the only
thing establishing that the JAX likelihood is right is that it reproduces the numpy forager's
per-trial choice probabilities from an independent implementation. If the JAX code starts
calling into the numpy dynamics for convenience, the two stop being independent and the parity
test degrades into asserting that a function equals itself.

This means the trial dynamics are deliberately written twice. That duplication is the point,
not an oversight, and a future contributor who "de-duplicates" it removes the guarantee.
