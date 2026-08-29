# Design note: hierarchical Bayesian (HB) cognitive-model baseline

**Status:** design, settled. All specifics in §7 are decided; implementation not yet started.
Written 2026-08-28 17:58 PT.

**One-line goal.** Reimplement the published hierarchical Bayesian Q-learning model in
NumPyro, extended with a population level, so it can be scored on held-out subjects on exactly
the same axis as the GRU/disRNN models and serve as the cognitive baseline for the project.

Cross-links: `docs/design-hierarchical-vi-foundation-model.md` §4 (the held-out matrix) and
§7 (why the HB and the foundation model are the same statistical object); ADR-0001, 0002, 0003.

---

## 1. Why this exists

The GRU/disRNN work needs a cognitive-model baseline that is scored identically, not
approximately. Two baselines already exist in the wrapper — `baseline_rl` (per-session MLE
via `aind-dynamic-foraging-models`) and the logistic-regression models — but neither does
partial pooling, so neither is the model the paper actually reports.

---

## 2. What the published model actually is

**Finding: the reference implementation is the hierarchical counterpart of `Hattori2019`.**
`aind_stan_fit_sim/code/stan_qLearning_5params.stan` is fit per subject by
`beh_1_load&fit.py:42` (`fit_animal(animalID)`, `N` = that subject's sessions), so it is
two-level with no cohort level.

Its parameters map onto the `Hattori2019` preset in `aind-dynamic-foraging-models`
(`ForagerQLearning`, `number_of_learning_rate=2`, `number_of_forget_rate=1`,
`choice_kernel="none"`, `action_selection="softmax"`) as follows:

| Stan (reference) | `aind-dynamic-foraging-models` | note |
|---|---|---|
| `aP` | `learn_rate_rew` | equivalent, see below |
| `aN` | `learn_rate_unrew` | equivalent, see below |
| `aF` | `1 - forget_rate_unchosen` | **inverted** |
| `beta` | `softmax_inverse_temperature` | bound differs: `[0,10]` vs `[0,100]` |
| `bias` | `-biasL` | **sign flipped** |

**Why the learning-rate mapping holds.** Stan branches on `sign(PE)`; the package branches on
`if reward:`. These coincide because `Q` is confined to `[0,1]`: it starts at 0, the chosen
update `(1-a)Q + a*r` is a convex combination with `a` in `[0,1]` and `r` in `{0,1}`, and the
unchosen update multiplies by a factor in `[0,1]`. Hence `r=1 => PE = 1-Q >= 0 => aP` and
`r=0 => PE = -Q <= 0 => aN`. The single tie (`r=0, Q=0`, where Stan takes the `aP` branch)
updates by `aP*0 = 0`, so it is a no-op.

**This equivalence breaks** if `Q` is ever initialised away from 0, or if reward becomes
non-binary. Anything that widens `Q` beyond `[0,1]` invalidates the mapping.

**The `aF` trap.** The reference calls `aF` a "forgetting rate", but it is a *retention*
factor — `aF=1` means no forgetting, the opposite of `forget_rate_unchosen=0`. See
`CONTEXT.md`; we use the package's sense throughout.

**On the priors.** The paper's "non-informative (uniform)" subject-level priors and the code's
`mu_p ~ normal(0,1)` are the same thing, not a contradiction: `Phi(X)` with `X ~ N(0,1)` is
exactly `Uniform(0,1)`, and `Phi(.)*10` is `Uniform(0,10)`, matching the published table's
`beta [0,10]`. The transform *is* the uniform prior.

Two genuine discrepancies do exist between the released code and the earlier paper's table:
`bias` is session-level and effectively flat (`normal(0,20)` on the logit scale) in the code
but subject-level `N(0,1)` in the table; and `sigma ~ cauchy(0, 0.2)` appears in neither paper's
table, though it is the single knob controlling how much session-to-session variation survives.

---

## 3. Decisions settled

- **NumPyro/JAX, not PyStan.** ADR-0001. Correctness anchor is a parity test against the
  numpy foragers, not against Stan.
- **Three-level: population -> subject -> session.** ADR-0002. `M=0, S=1` recovers the
  published model.
- **Held-out score is pointwise lppd, averaged in probability space.** ADR-0003.
- **Lives in `aind-dynamic-foraging-models`** as a subpackage with NumPyro behind an optional
  extra, so the dispatcher and wrapper get it through a dependency they already have
  (`baseline_rl_trainer.py` already imports `generative_model`).
- **Naming is `HB-<PresetAlias>`**, alias verbatim from `ForagerCollection.FORAGER_PRESETS`.
- **`bias` becomes hierarchical**, matching the earlier paper's table rather than the
  released code.
- **Two-stage empirical Bayes first**, full joint three-level fit as optional follow-up.
- **Recovery study before any real-data fitting.**

---

## 4. Model specification

Three levels, non-centred throughout (the hierarchy plus a half-Cauchy scale is otherwise a
textbook Neal's funnel):

    population:  mu_p_m ~ N(M, S)                      # per subject m, unconstrained
    subject:     theta_raw_{m,s} ~ N(0, 1)             # per session s, non-centred
                 theta_unc_{m,s} = mu_p_m + sigma_m * theta_raw_{m,s}
    session:     bounded params via Phi(theta_unc), unbounded params used directly
    likelihood:  choice_t ~ Bernoulli(logit = beta * (Q_R - Q_L) - biasL)

Transforms: rates and `choice_kernel_relative_weight` on `[0,1]` via `Phi(.)`; `threshold` on
`[-1,1]` via `2*Phi(.)-1`; `softmax_inverse_temperature` via `Phi(.)*B`; `biasL` unbounded,
so no transform — which is exactly the earlier paper's `bias ~ N(0,1)`.

Free parameters per family:

| model | parameters |
|---|---|
| `HB-Hattori2019` | `learn_rate_rew`, `learn_rate_unrew`, `forget_rate_unchosen`, `softmax_inverse_temperature`, `biasL` |
| `HB-Bari2019` | `learn_rate`, `forget_rate_unchosen`, `choice_kernel_relative_weight`, `softmax_inverse_temperature`, `biasL` |
| `HB-CompareToThreshold` | `learn_rate`, `threshold`, `softmax_inverse_temperature`, `biasL` |

`choice_kernel_step_size` is frozen at 1.0 for `Bari2019` by construction and is not free.

Sessions are padded to a common trial count with a validity mask, since JAX needs static shapes.

---

## 5. Recovery study protocol

**Generate with the numpy foragers, fit with the JAX HB.** Two independent implementations, so
a parameterisation error cannot cancel out — this is what catches the `aF` inversion and the
`biasL` sign flip automatically. Task is `CoupledBlockTask(reward_baiting=True)` from
`aind_behavior_gym`, already the idiom in that repo's `tests/test_Hattori.py`.

Ground truth must be generated from the **full three-level** structure, not from flat
per-session draws, or the study cannot detect the inflated-`S` failure mode in §6.

Metrics, in order of how much they matter:

1. **Per-parameter recovery** — posterior mean vs truth at session and subject level. These
   parameters are named and identifiable, so plain scatter/r-squared; no CCA needed, unlike
   the RNN latents in study 04.
2. **Coverage** — do 50%/90% credible intervals contain the truth at nominal rate? This is
   what MLE structurally cannot provide and is the main justification for the model.
3. **Variance-component recovery** — are `sigma_m` and `S` returned correctly? Most likely
   to be wrong, and where the two-stage trap shows up.
4. **Partial pooling beats per-session MLE** — fit the same simulated data with the package's
   existing `fit()` and show the gap widening as sessions shorten. This is the whole point of
   the model, measured against the tool already shipped.

Sampler gates: R-hat < 1.01, near-zero divergences, adequate ESS.

Prior-predictive check before any fitting: simulate from the prior and confirm behaviour is
plausible (not all sessions at chance or at ceiling). This also settles the `beta` cap
empirically.

Companion: **model recovery** — generate from each family, fit all three, confirm the
generating family wins on held-out likelihood.

---

## 6. Held-out evaluation

The metric is the wrapper's existing per-trial geometric-mean likelihood
(`exp(sum log p / n_trials)`, `baseline_rl_evaluation.py:117`), computed from a
`choice_prob[2, n_trials]` array. The HB emits that same array with posterior draws collapsed
per ADR-0003, so `_compute_normalized_likelihood` is reused unchanged and every model lands on
one axis.

**A held-out session's parameters were never inferred, and must not be inferred from that
session's own choices** — that is using the targets to fit the latent. Instead marginalise:

    draw (mu_p_m, sigma_m) from the subject posterior conditioned on context sessions
      -> draw a fresh theta_sess from that subject distribution
      -> score the held-out session
      -> average per trial in probability space

Report the whole matrix, not one cell:

| held out | infer from | isolates |
|---|---|---|
| trials within a session | earlier trials, same session | dynamics given session parameters |
| sessions within a training subject | that subject's other sessions | session-level pooling |
| held-out subject, zero-shot | nothing — population level only | quality of the population prior |
| held-out subject, few-shot (k) | k context sessions | adaptation to a new subject |

A single few-shot number confounds a good population prior with good adaptation. The headline
figure is the zero-shot -> few-shot(k) curve with GRU, disRNN, `baseline_rl` and HB on shared
axes.

**Two-stage caveat.** Fit the population to the subject-level posterior *draws*, not to
posterior means. Fitting to point estimates conflates each subject's posterior uncertainty with
genuine between-subject variance, inflating `S`, producing a too-diffuse population prior, and
making the HB look artificially weak at zero-shot.

**Do not** use the paper's 50-bin histogram-mode MAP for scoring; that is for comparing
parameters across models, and plugging a point estimate into held-out scoring favours whichever
model is more confident.

---

## 7. Settled specifics

All decisions below were settled 2026-08-28. Nothing in this note is open.

| # | Decision |
|---|---|
| 1 | Parameter conventions follow the package, not the reference Stan model (ADR-0004) |
| 2 | Population level pools **both** location and scale: `mu_p_m ~ N(M,S)` and `log sigma_m ~ N(m_ls, s_ls)` (ADR-0005) |
| 3 | `HB-Hattori2019` first; then `HB-Bari2019`; `HB-CompareToThreshold` last |
| 4 | Ignored trials excluded from the likelihood, no Q-update — matches `ignore_policy: "exclude"` |
| 5 | Session selection reuses the wrapper's `mature_only` / curricula / snapshot filters verbatim |
| 6 | NumPyro behind a `bayes` extra; core stays `>=3.9`; separate CI job on 3.11+ (ADR-0006) |
| 7 | Recovery lives entirely in `aind-dynamic-foraging-models` as a standalone suite depending on nothing in the disRNN stack, documented by a NumPyro tutorial notebook. The dispatcher study, `studies/08-hb-vs-gru-heldout/`, covers only the real-data held-out comparison against GRU/disRNN |
| 8 | Recovery pilot ~20 subjects x 20 sessions x 600 trials; full run ~100 x 40 |
| 9 | `softmax_inverse_temperature` cap configurable, default 10; check boundary pile-up in recovery |
| 10 | A `bayes` extra in the existing `pyproject.toml`, not a separate distribution |
| 11 | No half-Cauchy on `sigma`; pooling `log sigma_m` replaces it. Weakly-informative priors on the population scale (`m_ls ~ N(-1,1)`, `s_ls ~ HalfNormal(1)`), with `s_ls` swept in recovery |
| 12 | Runs through the wrapper's `ModelTrainer` interface like `baseline_rl`, sharing the data loader and W&B logging — not offline JSON |
| 13 | Few-shot grid is `k in {0, 1, 2, 4, 8}`, matching `studies/01-gru-scaling-law/heldout_fewshot_k*.yaml`; mirror the `_mature` variants at k=1 and k=4 |
| 14 | Subset reuses study 01's cohort exactly (`data.subject_ratio` against the ~614 pool, seed 0), so results compare against existing GRU numbers: D≈30 at `0.049` first, then D≈100 at `0.163` |
| 15 | Both estimators run on the subset; two-stage is only promoted to full scale if it matches one-stage. Threshold not yet fixed — the GRU's 0.0004 spread across seeds is the natural yardstick. **Amended 2026-08-29:** two-stage's *compute* rationale is gone — see the note below — so it now has to justify itself on statistics alone |
| 16 | Adaptation plugs in the population posterior mean (empirical Bayes), validated once against carrying full `p(M,S)` draws |
| 17 | Batched subject fits `vmap` the **sampler**, so each subject keeps its own step size and adaptation; one joint NUTS over all subjects would couple them and stop being two-stage. **Amended 2026-08-29:** deferring this was a mistake — it is what makes two-stage viable at all on GPU, not an optimisation |
| 18 | `chain_method` stays `vectorized` (the only way to batch on one GPU); the lockstep cost is measured by comparing ESS/draw at 1 vs 16 chains rather than assumed small |
| 19 | SVI is out of scope. Trigger to revisit: the one-stage joint fit failing to converge, which is a problem SVI solves and more compute does not |
| 20 | Sessions are never truncated. The GRU pads to `max_session_length` with `-1` masking and uses `length_bucketing`, changing compute but not the trial set, so the HB packs for the same reason and both score identical trials |

### Development setup

Core code is developed in `aind-dynamic-foraging-models`, checked out at
`/home/han.hou/code/aind-dynamic-foraging-models` on branch `feat/hierarchical-bayes`.

AGENTS.md §13's "HPC is pull-only runtime" rule is explicitly relaxed for this work: this is a
Claude Code session, not Claude Science, and the feature is developed on HPC directly.

Sessions run on a SLURM compute node (partition `aind`), so tests run locally here. Work
needing more CPU or a GPU is spawned to another compute node or to Beaker.

### Two-stage lost its compute rationale (2026-08-29)

Two-stage was adopted as the cheap, scalable approximation to a joint fit assumed to be
unaffordable. Measurement on the study 01 D≈30 cohort reversed that: the batched one-stage
fit took 3 h 32 m while the sequential two-stage fit was still running past 5 h 45 m.

The mechanism is the latency-bound behaviour recorded in
`aind-dynamic-foraging-models/benchmarks/RESULTS.md`. Wall time is set by scan **depth**, and
extra sessions are nearly free. A joint fit puts the whole cohort in one gradient and pays
that depth cost **once**; sequential per-subject fitting pays it **once per subject**.

So on GPU the joint fit is both the statistically preferred estimator and the cheaper one.
Two-stage now has to earn its place on statistics alone, and only at scales where the joint
fit fails to converge. The qualification is that this compares a batched implementation with
a sequential one — a two-stage fit that vmapped the sampler (decision 17) would be
competitive, which is precisely why deferring that work was a misjudgement.

Extrapolating flat lanes, a full-cohort joint fit at D≈614 should cost roughly what D≈30
costs plus a trajectory-length penalty from the higher dimension: order 8–12 hours, against
Stan's projected days to weeks. That figure wants the D≈100 rung before it is trusted.

### Cost structure of the k-sweep

The population fit is done **once** and cached; each `k` reuses it and only re-conditions the
held-out subjects. Refitting the cohort per `k` would multiply the expensive step by the size
of the grid for no gain. Sharing the wrapper's `DatasetBundle` is also what makes decisions 4
and 5 structural rather than aspirational — the alternative is reimplementing the trial and
session filters and hoping they match the neural models'.

---

## 8. Staging

Live status is tracked in issue #72, with per-repo work in
`aind-dynamic-foraging-models#62` and `aind-disrnn-wrapper#64`. This section holds only the
sequence and its verification gates, which are spec rather than state.

    1. JAX Hattori2019 likelihood
       -> verify: per-trial choice_prob parity vs the numpy forager
    2. NumPyro two-level model, per subject
       -> verify: M=0, S=1 reproduces the published per-subject model
    3. Population level over mu_p and log sigma, via two-stage empirical Bayes
       -> verify: S not inflated when fit to subject-level posterior draws
    4. Standalone recovery suite in aind-dynamic-foraging-models, plus a
       NumPyro tutorial notebook showing the results
       -> verify: parameter recovery, interval coverage, partial pooling beats
          per-session MLE at short session lengths
    5. Held-out eval, zero-shot + few-shot(k)
       -> verify: drops into _compute_normalized_likelihood unchanged

---

## Provenance

Distilled from the design discussion of 2026-08-28, which established the `Hattori2019`
equivalence and its two parameterisation traps by reading
`AllenNeuralDynamics/aind_stan_fit_sim` against
`aind-dynamic-foraging-models` v0.13.0. Method text from the published paper and from the
earlier bioRxiv version (2022.12.08.519670) supplied the prior specifications.
