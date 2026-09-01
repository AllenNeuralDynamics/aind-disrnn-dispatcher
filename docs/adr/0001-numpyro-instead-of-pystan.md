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

## Amendment, 2026-08-29: the speed rationale did not survive measurement

Benchmarked against the reference PyStan implementation on identical synthetic data
(40 sessions x 650 trials, 16 chains). Full numbers in
`aind-dynamic-foraging-models/benchmarks/RESULTS.md`.

**Stan is 3.7x faster than our best NumPyro configuration and 5.7x faster than an A100.**
An A100 is slower than a 2017 TITAN Xp, which is the signature of a latency-bound workload:
each gradient is 650 sequential scan steps over only ~640 lanes doing scalar arithmetic, so
wall time is dispatch overhead that GPU generation does not improve. A sweep of sampler
geometries found no fix. Stan also loops to each session's true length, while JAX must pad
to the maximum, so real long-tailed session data will widen the gap further.

The claim in this ADR that the trial loop "expresses cleanly as a `lax.scan`" is true and
irrelevant; that it would therefore be fast was asserted, not measured, and was wrong.

**What survives.** Two of the original reasons are untouched: the three-level cohort model
and SVI are both NumPyro-native, and neither is what Stan is good at. A third has appeared
and is the decisive one: because two-stage subject fits are *independent*, every subject can
occupy one vmapped computation. A latency-bound workload widens from ~640 to ~10,000 lanes
almost for free, which would make the whole cohort about as cheap as one subject is today --
roughly 40 minutes against Stan's projected 23 hours on 128 cores. That is unmeasured.

**Status.** This decision is now conditional. If cross-subject batching delivers, NumPyro
wins the case it was chosen for. If it does not, the honest resolution is a split: Stan for
production per-subject fits, NumPyro for the joint three-level model and SVI. Note the
tension this exposes -- two-stage empirical Bayes fits per subject, which is precisely
Stan's favourable regime, so the architecture and framework choices are currently pulling
against each other.

## Resolution, 2026-08-29: the conditional is met, NumPyro stays

The batching experiment ran. Widening one gradient from 640 to 20,480 lanes at fixed depth
costs 1.0x the time on an A100 -- 0.03 of linear, with per-lane cost falling 33x
(`aind-dynamic-foraging-models/benchmarks/RESULTS.md`). The workload is latency-bound on GPU,
so batching across subjects is nearly free and the whole cohort costs about what one subject
costs today. The pre-committed threshold was 2x; the result is 1.0x.

This also explains the amendment above rather than contradicting it. Both benchmarks ran at
640 lanes, where an A100 sits at a few percent utilisation, so they measured dispatch
overhead rather than compute -- which is why a newer GPU was slower than an older one. **The
measurement was sound; the configuration was wrong.** Stan really is faster for a single
subject fit at a time. That is simply not the configuration to run.

**Consequence: batching is not an optimisation here, it is the decision.** Fitting subjects
one at a time forfeits the entire rationale for this ADR and loses to Stan by 3.7x. The
per-subject code path should be treated as a development convenience, never a production one,
and the same applies to the one-stage joint fit, whose cohort-wide gradient is what the flat
regime was measured on.

Untested: whether flatness extends past ~20k lanes. A 16-chain cohort fit needs ~400k lanes.
