# Plan: HB production readiness, then the data-scaling ladder

**Status:** active plan. This file is **spec** — the gates, their acceptance criteria, and
the ladder design. **Live state is not here:** it is dispatcher issue #107 (this push) and
issue #72 (the HB programme). A status line in this file would be a second copy that rots,
so there is none.

Written 2026-09-02, from four gates set by Han before production rungs may launch.

Cross-links: `../../docs/design-hb-baseline.md` (settled decisions — read §6 for the
held-out matrix and §7 for the 20 settled specifics), `README.md` (cohort + comparators),
`variants/one-stage-ladder/notes.md` (what the variant is).

---

## 1. Why gates at all

Two things landed back-to-back and had never been exercised together: the
`disrnn` → `dynamic-foraging-bfm` identity rename (ADR-0007) and the HB integration
(models#63, wrapper#65). The first HB job on Beaker is therefore also the first job of any
kind on the renamed stack.

The programme's own record is the argument for gating rather than trusting a green exit.
Issue #72's item 0 puts it plainly: four separate times in this work something passed
locally and failed in the place it actually runs. And this project has already been bitten
by the specific failure that a gate catches and an exit code does not — a run that exits 0
with the held-out metric silently absent, invisible from either Beaker or W&B alone.

So: **no production rung launches until gates 1–4 are resolved.** A gate is resolved when
its acceptance criterion below is met by a fetched artifact, not by a log line.

---

## 2. Gate 1 — every useful result is in W&B and comes back out

The HB fit is the expensive step; everything downstream (a new rung, a calibration check,
a figure) is supposed to reuse it rather than refit. That only holds if the posterior is
genuinely retrievable, which is a stronger claim than "the upload succeeded".

**Must be present and re-readable for a completed rung:**

| item | W&B location |
|---|---|
| matched-conditioning held-out likelihood | summary `heldout/eval_likelihood`, `heldout/test_likelihood`, `heldout/matched_likelihood` |
| few-shot rungs | summary `heldout/few_shot_k{0,1,2,4,8}_likelihood` |
| cohort size actually scored | summary `heldout/num_test_trials`, `heldout/num_test_subjects` |
| per-subject detail | table `heldout/per_subject_matched` |
| posterior draws + diagnostics + provenance | artifact `hb-fit-<estimator>-D<N>` → `<estimator>_fit.nc` |
| launch provenance | run config `meta.{study,variant,launch_id,label,note,config_hash}`, plus `wrapper_commit` / `dispatcher_commit` / `foraging_models_commit` |

**Acceptance criteria.**

1. Every key above is fetched back through the W&B API — not read off the job log.
2. The `.nc` artifact is downloaded and reopens as InferenceData, with the posterior group
   and the sampler diagnostics both non-empty. An artifact that uploads but will not reopen
   is exactly what this gate exists to catch.
3. The per-subject table's row count equals `heldout/num_test_subjects`.
4. The resolved seed is present in the fit artifact, so a rung is attributable even if it
   predates the seed fix (#108).

**Why criterion 3 is not pedantry.** The held-out loader already drops subjects between
selection and fetch — 153 selected, 149 fetched on the D≈10 smoke, with four subjects
passing the `>=min_sessions` filter but contributing no trials. A summary count that
disagrees with the table is how that turns from a known data condition into a silent
denominator error in a reported likelihood.

---

## 3. Gate 2 — diagnostics and presentation figures, from a real fit

`aind-dynamic-foraging-models/src/aind_dynamic_foraging_models/hierarchical_bayes/plotting.py`
exists, and this folder already carries `figures/population_recovery.png`,
`figures/conditioning_curve.png`, `figures/shrinkage.png`. Those are **synthetic**, with
known ground truth — correct for validating the plotting code, wrong to read as results
(#72 tracks their regeneration as an open box).

Unverified as of writing, and the first thing to establish: whether a wrapper HB run logs
any figure to W&B at all. If it does not, wire it.

**Required set.** Split by what each is *for*, because the two audiences differ:

*Sampler trustworthiness* — these decide whether the numbers may be quoted:
- trace and rank plots per population parameter
- energy / divergence summary; `r_hat` and ESS per parameter against the §5 gates in the
  design note (R-hat < 1.01, near-zero divergences)

*Scientific reading* — these go in a talk:
- per-parameter population posterior densities, on the interpretable scale (post-`Phi`),
  not the unconstrained one
- shrinkage: per-subject posterior means against their per-session MLE counterparts, which
  is the partial-pooling claim made visible
- the zero-shot → few-shot(k) conditioning curve, with GRU / disRNN / `baseline_rl` on
  shared axes — the design note calls this the headline figure

*Trajectory (feasibility to be established, not assumed).* The HB analogue of a latent
trajectory is the per-session latent `Q_R - Q_L` and the implied choice probability, drawn
**with a credible band** across posterior draws. That band is the thing per-session MLE
structurally cannot produce, so it is the most honest single picture of why the model is
Bayesian. Whether it can be emitted without surgery depends on whether `likelihood.py` can
return per-trial `Q` alongside the choice probabilities. **If it cannot, say so and stop** —
do not reshape the likelihood to produce a figure.

**Acceptance criteria.** The sampler-trustworthiness set is logged for the rung being
gated, and each scientific figure is either logged from real data or recorded here as
infeasible with the reason. Committed synthetic figures are relabelled so they cannot be
mistaken for results.

---

## 4. Gate 3 — surface the inconsistencies

Findings from this push. Status lives on the board, not in this table; the entries without
an issue are the ones still to triage.

| # | Finding | Where |
|---|---|---|
| 1 | HB ignored `seed=` and seeded NUTS from wall-clock time | #108 |
| 2 | Resumable launcher silently records `dispatcher_commit: null` from a sandbox checkout | #109 |
| 3 | 4 held-out subjects pass selection but contribute no trials (153 → 149) | triage |
| 4 | The memory-adaptive session-chunk size is never logged, so a run's occupancy cannot be read back | triage |
| 5 | `arviz` shape-validation flood at `num_chains=1`; whether `max_r_hat` is meaningful there | triage |
| 6 | Run displayName carries `multisubject` twice | triage |
| 7 | W&B group naming split: HPC's `hb-one_stage@…` vs the launcher's `<variant>@<launch_id>` | triage |
| 8 | jax 0.6.1 (Beaker image) vs 0.10.2 (HPC HB envs) — same seed need not give the same posterior | #101 |

Finding 8 is the one with teeth for this study: an HB arm and a GRU arm compared across
different JAX versions is an uncontrolled variable in the comparison, and #101 records that
no one has decided which stack is the reference. The Beaker runs are the first data point
on the wrapper-pinned side.

**Acceptance criterion.** Each row is filed with a root cause, and anything that can change
a reported number is fixed — not merely filed — before the ladder launches.

---

## 5. Gate 4 — is the fitting speed sensible

**Do not compare across incomparable axes.** The D≈10 smoke's 658 s population fit ran at
30 warmup / 30 samples / 1 chain; the design note's D≈30 one-stage 3 h 32 m ran a different
cohort at 500/500/4 on a different GPU. Neither predicts the other, and quoting one for the
other is the specific error this gate forbids.

**Compare against the model the repo already measured.** `hierarchical_bayes/benchmarks/`
ships `RESULTS.md`, `lane_scaling.py`, `geometry_experiment.py`,
`benchmark_stan_vs_numpyro.py` and `reference_stan/stan_qLearning_5params.stan`. So both
comparisons Han asked for are measurable rather than hypothetical:

1. **Against the previous HPC run** — same estimator and cohort dial, different device and
   JAX stack (finding 8). Isolate the variables before attributing any difference.
2. **Against PyStan** — the reference model and a benchmark script are both in-tree. The
   design note's projections ("order 8–12 h at D≈614", "Stan's projected days to weeks")
   are extrapolations; establish which are measured and which are not, and label them
   accordingly per AGENTS.md §11.

**The prediction to test.** The recorded mechanism is that this workload is *latency-bound*:
wall time is set by scan **depth** (trial count), while extra lanes (subjects × sessions)
are nearly free. That is what makes a joint fit cheaper than sequential per-subject fitting —
depth is paid once instead of once per subject. So the D≈30 rung should cost roughly what
D≈10 costs at equal sampler settings, plus a trajectory-length penalty from the higher
dimension. **A rung whose wall time scales with subject count instead falsifies the
mechanism**, and that finding matters more than the timing itself.

**Acceptance criterion.** A measured wall-clock breakdown (population fit, adaptation,
session scoring) for the gated rung, the observed session-chunk size (finding 4), and a
stated verdict on whether the latency-bound model holds — with "verified" and "likely,
unconfirmed" labelled separately.

---

## 6. Then production — the data-scaling ladder

Mirror study 01's D dial so the results land on the curve `analyze_scaling.py` already
reads, into project `mice_data_scaling`:

| `data.subject_ratio` | D | GRU comparator on record (seeds 0/1/2) |
|---|---|---|
| 0.016 | ≈10 | — |
| 0.049 | ≈30 | 0.7248 / 0.7252 / 0.7250 |
| 0.163 | ≈101 | 0.7264 / 0.7264 / 0.7260 |
| 0.489 | ≈300 | — |
| 1.0 | ≈614 | — |

Other comparators already on record: per-mouse MLE Hattori 0.71267, per-mouse MLE CTT
0.71704. The GRU's ~0.0004 spread across seeds is the yardstick for whether an HB–GRU
difference means anything.

**Walk the ladder upward; do not fire all five rungs at once.** Flat lane scaling is
verified only to ~20k lanes, and the full cohort needs roughly 20× that. If it saturates,
D≈614 will not behave like the smaller rungs — so D≈614 is its own decision, taken after
the D≈300 rung's timing is in hand, not a fifth item in a fan-out.

**Launch mechanics.** Through `launch_beaker_resumable.py` with
`variants/one-stage-ladder/sweep_beaker.yaml` + `experiment_beaker.yaml`. Two properties of
that template are load-bearing and should not be "simplified" later:

- `{priority: normal, preemptible: false}`, against the launcher's low/preemptible burst
  default. `hb_hattori.yaml` has no `training` block, so there is no
  `checkpoint_every_n_steps` and nothing for `autoResume` to resume from — a preemption
  restarts the NUTS fit at zero.
- An S3-reachable cluster only. `mice_snapshot_scaling` reads the AWS parquet cache, which
  the GCP pools cannot reach however many GPUs they show free.

**Seeds.** The GRU curve carries three seeds per rung. Whether HB needs the same depends on
its own seed spread, which is unmeasured — and was unmeasurable before #108, since the seed
was not controlled. Measure the spread at one rung before deciding, rather than tripling the
whole ladder on the assumption that it matches the GRU's.
