# Variant rl-baseline-hattori

**Agent.** Hattori 2019 — `ForagerQLearning`, **2 learning rates** (asymmetric $\alpha^+/\alpha^-$), 1 forget rate, no choice kernel, softmax (`L2F1`).

**Fitted parameters.** `learn_rate_rew`, `learn_rate_unrew`, `forget_rate_unchosen`, `biasL`, `softmax_inverse_temperature`
*(Verified by constructing the agent, not read off a doc.)*

**Goal.** Two things at once:

1. **Extend r8's baseline comparison (issue #20).** Study 01's r8 showed the GRU beats a
   per-mouse **Bari** fit by +0.0136 at D=614 (100% of held-out mice, p~3e-26). That is a
   one-model baseline. Fitting Bari + Hattori + CTT turns it into *"the GRU beats the best
   of three classical RL models"*, which is a far harder claim to dismiss.
2. **Supply `model2` for the embedding analyses (issues #24 / #27).** `run_analysis
   embedding-params` and `likelihood-advantage` need per-subject fitted RL parameters for
   the **training** mice — the only mice with a learned GRU subject embedding.

**Why this variant exists at all (vs `rl-baseline-simple`).** That variant ran with
`model.heldout_refit.skip_train_fit=true`, so it **never fit the training subjects** — it
fit and scored only the 149 reserved held-out mice. Its parameters are therefore unusable
for any embedding analysis: RL params exist for mice that have no embedding, and embeddings
exist for mice that have no RL params. Setting `skip_train_fit=false` (the config default)
fits **both** cohorts in a single run.

Never fit at this scale. Issue #20 lists it as in-scope; the only prior fit was on the legacy 10-mouse Code Ocean `train10` cohort.

## Cohort (verified by composing the config, not assumed)

Identical to study 01's D=614 arm, so the comparison is arm-for-arm:

| knob | value | meaning |
|---|---|---|
| `subject_ratio` | 1.0 × 3 curricula | the full ~614-mouse training pool |
| `heldout_every_n` | 5 | the same fixed 149-mouse held-out cohort as the GRU |
| `snapshot` | 20260603 | same pinned DB snapshot |
| `eval_every_n` | 2 | same per-subject train/eval session split |
| `ignore_policy` | exclude | 2-way L/R (study 01's target, not study 02's 3-way) |

## Outputs

- **train cohort** → per-subject fitted params (the `model2` artifact).
- **heldout cohort** → refit + scored under `heldout/*`, matching the GRU namespace
  (`heldout/eval_likelihood` is the sweep metric), so it drops straight into r8.

## Compute

CPU-only differential evolution, parallelized **over subjects** (`multisubject_subject_workers=112`,
`DE_kwargs.workers=1` to avoid nested oversubscription). One 112-core `cpu`-featured node
(the 6238R class; 16 were idle at launch time). `aind` max walltime is 21 days, so the
72 h request carries no walltime risk.

**Launch:** see the header of [`sweep.yaml`](sweep.yaml).
