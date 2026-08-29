# NumPyro instead of PyStan for the hierarchical Bayesian baseline

The published cognitive model and its reference implementation
(`AllenNeuralDynamics/aind_stan_fit_sim`) are written in Stan. We are reimplementing it in
NumPyro/JAX rather than porting the Stan model, because the trial loop expresses cleanly as a
`lax.scan`, sessions vectorise under `vmap`, chains run vectorised on one GPU, and it shares
the JAX stack the GRU/disRNN models already use. It is also the only way the three-level cohort
model (ADR-0002) is affordable, and it gives SVI for free, which connects to
`docs/design-hierarchical-vi-foundation-model.md`.

## Consequences

We give up the published implementation as a correctness oracle. The replacement anchor is a
parity test against the numpy foragers in `aind-dynamic-foraging-models`: same parameters and
same choice/reward history must produce the same per-trial choice probabilities. This is
arguably stronger than Stan parity, because the two implementations are independent, so a
parameterisation error cannot cancel out — see the `aF` inversion and `biasL` sign flip
documented in `docs/design-hb-baseline.md`.
