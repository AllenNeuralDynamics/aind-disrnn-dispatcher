# Design note: hierarchical (mixed-effects) foundation model via amortized VI

**Status:** design / plan (not yet implemented). Forward-looking spec for a future
modeling direction. Written 2026-07-13.

**One-line goal.** Turn the multisubject sequence model into an explicit
**hierarchical random-effects model** (population → subject → session) trained by
**amortized variational inference** (no MCMC), so it (a) captures individual *and*
session-wise heterogeneity with principled partial pooling, (b) yields per-session
posterior *uncertainty*, and (c) lines up arm-for-arm with the cognitive
hierarchical-Bayesian (HB) models on the same synthetic ground truth.

---

## 1. Motivation — what study 04 (Stage-4b) told us

Study `04-gru-vs-disrnn-embedding-recovery`, variant `gru-stage4b-concentration`
(Dirichlet concentration sweep, group `@20260712-060852`) established:

- **Session conditioning is a smooth function of session *position*, not content.**
  The only session-encoding modes are `{none, scalar, fourier}`
  (`code/models/session_conditioning.py`), all of which feed a delta-MLP from the
  normalized **session index**. Confirmed empirically: the session-conditioned
  embedding predicts session *position* at R² ≈ 0.58–0.66 but adds **zero** over the
  subject embedding for decoding the realized per-session **family** (gap ≈ 0 at
  every concentration 0.5 → 5).
- **Therefore the `gru-stage4b` null ("session conditioning adds nothing") is
  robust, not a sparsity artifact.** A continuous, position-indexed session code is
  structurally unable to represent a discrete per-session regime draw. It is the
  right inductive bias for smooth *drift* (Stage-2) and the wrong one for discrete
  *switching* (Stage-4b).

The reframe this note builds on: the current model is **already a crude two-level
mixed-effects model** — shared dynamics (fixed effect) + subject embedding table
(subject random effect, L2-shrunk) + session delta (session random effect,
`lambda_reg_session`-shrunk, but smoothness-constrained and MAP point-estimated).
This note makes that structure explicit, Bayesian, and flexible.

### The pooling spectrum (where each current mode sits)

| session mode | pooling analog |
|---|---|
| `none` (subject embedding only) | **complete pooling** across a subject's sessions |
| `scalar` / `fourier` (+ `lambda_reg_session`) | **partial pooling** with a *smooth position prior* + shrinkage |
| *free / content-inferred per-session latent* (not built) | **no pooling** = per-session fit (like fitting a cognitive model per session) |

There is **no** free-per-session or content-inferred session latent today. Adding
one — under a hierarchical prior — is the core of this design.

---

## 2. Design A (minimal): hierarchical Gaussian random effects

Keep the shared dynamics; make the latents a two-level Gaussian hierarchy with
**learned** variances:

```
population:  μ                       (learned global mean)
subject:     e_s      ~ N(μ,   Σ_subj)     # subject random effect
session:     e_{s,t}  ~ N(e_s, Σ_sess)     # session deviation from its subject
likelihood:  y_{s,t}  ~ RNN_θ( · | e_{s,t})  # session latent conditions the dynamics
```

Learnable: `θ` (shared dynamics), `μ`, `Σ_subj`, `Σ_sess` (diagonal, softplus),
plus the posterior (Section 3). `Σ_sess / Σ_subj` are the **variance components** —
the FM analog of the ICC / random-effect variances an HB reports (how much
heterogeneity lives at each level).

**Why minimal vs. today:** we already have `e_s` (the subject embedding table) and a
session-conditioning injection slot. Three edits:
1. `e_{s,t}` becomes a *free sampled latent* instead of `delta(session_index)` —
   drops the smooth-position constraint.
2. add the `N(e_s, Σ_sess)` link so sessions shrink to their subject.
3. replace the fixed `lambda_reg_session` L2 with a *learned* prior (the KL term below).

Stay Gaussian (graded) for the minimal version — this is the correct model for
smooth heterogeneity (drift, engagement), which is what real mice most plausibly
have. The discrete-regime extension is Section 5.

**Degenerate check:** if the posterior is a point mass and `Σ` is fixed, the whole
thing reduces to today's `max-likelihood + L2`. The current model *is* this
framework with uncertainty and learned variances switched off — we are turning
knobs back on, not replacing the model.

---

## 3. Training without MCMC: amortized variational inference

### 3.1 What we're inferring

Each session has a posterior `p(e_{s,t} | that session's choices & rewards)` — a
*distribution* over the session's latent given its ~650 trials. Classical HB gets
this per session via MCMC (samples). We instead **optimize** an approximate
posterior `q`, turning inference into gradient descent on the same
autodiff/GPU/minibatch stack the RNN already uses.

### 3.2 The ELBO

For a lower bound on `log p(data)`:

```
log p(y) ≥  E_q[ log p(y | e) ]  −  KL( q(e) ‖ p(e) )   =  ELBO
              └─ fit the data ─┘     └─ shrink q toward the prior ─┘
```

Maximizing the ELBO fits choices *and* pulls each latent toward its hierarchical
prior — **the KL term is the partial pooling.**

Full **hierarchical ELBO** (two levels):

```
ELBO = Σ_{s,t} E_q[ log p(y_{s,t} | e_{s,t}) ]
     − Σ_{s,t} KL( q(e_{s,t}) ‖ N(e_s, Σ_sess) )   # session shrinks to subject
     − Σ_s     KL( q(e_s)     ‖ N(μ,  Σ_subj) )   # subject shrinks to population
```

Optimize over `θ, μ, Σ_subj, Σ_sess`, and the inference params — all by Adam,
doubly-stochastic (minibatch over (subject, session) + sample the latent). No MCMC,
no per-unit sampler; scales to thousands of sessions.

### 3.3 Reparameterization (what removes MCMC)

To backprop through `e ~ N(m, s²)`, write `e = m + s·ε`, `ε ~ N(0, I)`. Now `e` is a
differentiable function of `(m, s)`, gradients flow, and it is plain SGD.

### 3.4 Amortization (what makes it scale + generalize)

**The idea.** "Amortize" = spread a one-time cost over many uses. Instead of solving
a fresh inference problem per session (MCMC or per-session VI), train **one shared
neural network — the encoder / inference network `φ`** — that reads a session's data
and directly emits that session's posterior:

```
session (choice, reward) sequence  ──►  encoder φ  ──►  (m, s²)   giving   q(e_{s,t}) = N(m, s²)
```

Inference for any session is then a **single forward pass**. Pay a big cost once
(training `φ`); every session's posterior is then nearly free, and — because `φ` is
shared — it **generalizes to sessions never seen** (feed a held-out session's data,
get its posterior; this is few-shot/in-context inference).

**Analogy.** Per-session MCMC = hiring a statistician to hand-fit each session from
scratch. Amortization = training an intern on thousands of sessions until they can
glance at a new session's behavior and immediately write down the posterior. In ML
terms: instead of *solving* an optimization per session, you *learn the function*
"data → posterior."

**"Learn the posterior from the session" means:** the posterior is computed from
that session's own observations by the shared encoder. A session whose behavior
looks lose-shift → `m` lands in the LossCounting region; ambiguous behavior → wide
`s²`. The encoder's shared weights encode the *general* rule "this behavioral
signature ↦ this region of latent space, with this confidence," learned from the
whole dataset — which is exactly what a fresh MCMC run cannot reuse or transfer.

The encoder is trained **jointly** with the model by maximizing the ELBO; the
reparameterized sample lets the prediction error backprop through the latent into
`φ`, so the inference rule is learned end-to-end (no separate inference algorithm).

### 3.5 The inference spectrum (place it against MCMC)

| method | inference cost | new session? | quality |
|---|---|---|---|
| **MCMC per session** | slow, per-session | re-run from scratch | ~exact |
| **per-session VI** (non-amortized) | one optimization per session; params grow with #sessions | re-run | good, per-unit |
| **amortized VI** (encoder) | one forward pass; constant params | **instant** | approximate (amortization gap) |

### 3.6 The honest catch: the amortization gap

One shared network can't be perfectly tailored to each session, so amortized `q` is
looser / more miscalibrated than exact inference. Fixes when it matters (e.g. the
final calibration comparison vs. MCMC-HB): a few SVI steps refining the encoder's
output per unit ("semi-amortized"), a more expressive `q` (normalizing flows), or
importance weighting (IWAE). For everyday training + identifiability diagnostics the
plain encoder is fine.

---

## 4. Evaluation & model comparison (held-out validation)

In a hierarchical latent model, "held-out" is **one split per level of the
hierarchy** — a matrix, not a scalar. The matrix *is* the comparison.

| hold out… | infer the latent from… | tests | existing knob |
|---|---|---|---|
| **trials** within a session | that session's *other* trials | dynamics given the session latent | (trial split) |
| **sessions** within a subject | the subject's *other* sessions | subject structure generalizes across sessions | `heldout_session_mode=tail`, `heldout_frac` |
| **subjects — zero-shot** | nothing (`e = μ`) | shared dynamics + population mean | held-out cohort |
| **subjects — few-shot** | *k* sessions of the new subject | is the latent a usable coordinate for a *new* animal | `auto_heldout_finetune` / `heldout_refit` |

**Golden rule (validity for latent-variable models):** infer the latent from
*context only*, score *disjoint* targets. Never let the encoder see the held-out
targets. Amortized `q(e | context)` makes this clean — it reads only the context you
allow.

**Metric (class-agnostic currency):** posterior-predictive held-out log-likelihood,
marginalizing latent uncertainty:

```
log p(y* | context) = log ∫ p(y* | e) q(e | context) de  ≈  log (1/L) Σ_l p(y* | e^(l)),  e^(l) ~ q(e|context)
```

- **Marginalize, don't plug in the MAP** — scoring at `e = m` overstates fit and
  unfairly favors the more-confident model.
- **Normalize to the ground-truth policy** → the existing
  `likelihood_relative_to_groundtruth` (ceiling 1.0), so FM-VI, cognitive-HB, and
  `baseline_rl` all land on one axis, per held-out cell.

**Bayesian evidence (optional corroboration):** ELBO is a biased evidence proxy
(bound tightness varies across models); tighten with IWAE. **WAIC / PSIS-LOO** port
directly from the HB world — computed from pointwise held-out log-likelihoods + their
spread across posterior draws, which VI supplies via reparam sampling. Recommendation:
**held-out predictive LL primary, PSIS-LOO (`k̂` diagnostic) corroborating.**

**Second axis on synthetic ground truth — calibration.** Predictive LL says which
model *predicts* better; ground truth lets us ask which is *right*:
- recovery: CCA of posterior means to true params (existing metric);
- coverage: do the credible intervals cover the true `e` at their stated level? This
  is where amortized VI can lose to MCMC-HB at equal predictive LL (amortization gap
  + posterior collapse → over-tight `q`). Same check runs on the cognitive HB, so
  they are directly comparable.

---

## 5. Extension (Design B): discrete regimes

If we decide real behavior has **discrete regime switches** (strategy flips, state
changes), a continuous session code is the wrong inductive bias. Two ladder rungs:

1. **Minimal — content-inferred *discrete* session code.** Replace the position
   delta with a small session encoder → logits over K states, discretized via
   **Gumbel-softmax** (temperature-annealed soft→hard) or **VQ** (codebook of K
   vectors). Keeps the model; the code index becomes the recovered regime.
2. **Full — mixture-of-experts.** K expert dynamics modules + a **content-based
   gating network** (reads the session's trajectory) → soft/hard assignment over
   experts. `p(choice) = Σ_k g_k(session)·expert_k(trial)`. The gate assignment *is*
   the recovered per-session regime; each expert is an interpretable regime model.

In the Bayesian framing these are just a **prior-family choice on the session
latent**: Gaussian = graded (Design A); **mixture / Dirichlet-process = discrete
regimes (Bayesian MoE)**. The subject-level mixture weights are the subject's random
effect over regime propensity — exactly the Stage-4b generative structure.

**Gotchas:** gate/encoder input must be *content*, not position (else you rebuild the
current failure); `K ≥` #true regimes with a **load-balancing** loss (avoid expert
collapse); soft trains easily but blurs regimes, hard (Gumbel) is crisper but needs
temperature annealing; i.i.d. regimes (Stage-4b) → plain per-session gate, *not* an
HMM prior (that would be a mismatched sticky prior). Identifiability ceiling: if
~650 trials don't *behaviorally* distinguish the families, no mechanism recovers them
— an MoE won't rescue what the data doesn't contain.

---

## 6. Mapping to the existing code (the diff)

- **subject table** (`subject_embedding_size`, multisubject) → `q(e_s)`; start with
  per-subject variational params `(m_s, log s_s²)` — the current table plus a
  log-variance column — before adding a subject encoder.
- **session delta-net** (`code/models/session_conditioning.py`, modes
  `none/scalar/fourier`) → a small session encoder (MLP or tiny GRU over the
  session's trials) emitting `(m, log s²)`; sample via reparam; inject at the
  existing `session_integration_type` point (`direct` / `pre_mlp`).
- **`lambda_reg_session`** (fixed L2 "session-delta regularization") → the
  `KL(q_sess ‖ N(e_s, Σ_sess))` term with **learned** `Σ_sess`; add the subject-level
  KL. Make `Σ_subj, Σ_sess` learnable (softplus).
- **KL warmup** → reuse the existing session-conditioning λ-forward schedule
  (`session_curriculum_lambda`, pretrain → ramp) as **β-annealing** on the KL, so the
  model learns to *use* the latent before the prior clamps it. This is the must-have
  against **posterior collapse** (the VAE cousin of MoE expert-collapse); add
  **free-bits** if a dim still collapses.
- **held-out** → reuse `heldout_session_mode=tail` (held-out sessions) and the
  few-shot finetune split (`auto_heldout_finetune`) — but swap the current
  *gradient-fine-tune* inference for **encoder-inferred `q` + posterior-predictive
  scoring**. Metric stays `likelihood_relative_to_groundtruth`, so downstream
  reporting is unchanged.

---

## 7. Arm-for-arm with the cognitive HB

The cognitive HB and this FM are the **same statistical object** — hierarchical
random effects over a per-unit latent — differing only in whether that latent is the
**hand-specified cognitive parameters** (HB) or a **learned RNN latent** (FM). The FM
trades interpretability for flexibility (learns the dynamics); everything else —
partial pooling, variance components, posteriors, graded-vs-mixture priors — ports
over. On shared synthetic ground truth, compare them on identical axes:
**recovery** (CCA of posterior means to truth) and **calibration** (coverage of the
posteriors), per held-out cell, with `baseline_rl` as the correct-model-class
reference.

---

## 8. Suggested staging

1. **Stage 5a — minimal Gaussian hierarchical (Design A).** Learned `Σ`, free session
   latent, amortized encoder, KL β-warmup. Validate on Stage-1/2 synthetic
   (recovery + calibration should match or beat the current point-estimate model),
   then Stage-2b (drift). Deliverable: recovery + coverage vs. the existing GRU and
   vs. cognitive HB.
2. **Stage 5b — discrete session code (Design B, minimal).** Gumbel/VQ session code;
   re-run the Stage-4b concentration sweep. Success = the code index recovers the
   per-session family where continuous conditioning gave gap ≈ 0.
3. **Stage 5c — mixture-of-experts (Design B, full).** Only if 5b shows a discrete
   code helps and interpretable per-regime experts are wanted.

## 9. Open questions / risks

- **Posterior collapse** — the central VI failure; needs β-warmup + free-bits, and a
  monitor on latent usage (KL per dim).
- **Amortization gap** — encoder `q` looser than exact; close it (semi-amortized /
  IWAE) only for the final calibration comparison.
- **Latent identifiability** — RNN latent unidentified up to rotation/scale; recovery
  scored by CCA-to-truth, not raw coordinates (reuse study-04 tooling).
- **Is discrete needed?** Design B is a real architecture change; only pursue if we
  believe real behavior has discrete regime switches. If within-subject variation is
  graded (drift/engagement), Stage-2 already covers it and Design A suffices.

## Provenance

Distilled from the study-04 Stage-4b concentration-sweep result
(`studies/04-gru-vs-disrnn-embedding-recovery/variants/gru-stage4b-concentration/`)
and the design discussion that followed it (session 2026-07-12/13). Cross-link:
that variant's `notes.md` and the study `FUTURE_DIRECTIONS.md`.
