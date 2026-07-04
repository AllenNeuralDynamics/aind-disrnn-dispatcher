# Study: embedding-recovery

**Scientific question.** When a data-driven sequence model (GRU, later disRNN) is
trained on multisubject foraging data, do its learned **subject embeddings** (and,
in Stage 2, its **session conditioning**) actually recover the *known* generative
structure of the animals — or are they arbitrary codes that happen to fit?

We answer this on **synthetic data with ground truth**: each synthetic "subject"
is a Q-learning agent occupying a distinct subregion of parameter space, so we
know the true per-(subject, session) parameters and the true generating policy.

## Ground-truth generative model

Data is produced on-the-fly by `HierarchicalCognitiveAgents`
(`aind-disrnn-wrapper/code/data_loaders/hierarchical_synthetic.py`,
Hydra `data=synthetic_hierarchical`). It is **deterministic** given a seed
(instance-RNG task + agent, non-overlapping seed hierarchy), so no frozen dataset
is needed — every run regenerates byte-identical data and writes a
per-(subject, session) ground-truth parameter table (`groundtruth_params.csv`).

- **Agent:** `ForagerQLearning` — `number_of_learning_rate=1`, `number_of_forget_rate=0`,
  `choice_kernel=none`, `action_selection=softmax`.
- **Recoverable generating parameters (the dim-3 ground-truth embedding):**
  `learn_rate` ~ U[0.1, 0.9], `biasL` ~ U[-1.5, 1.5],
  `softmax_inverse_temperature` ~ U[2, 15]. Each subject = one centroid.
- **Stage 1 (this scaffold):** parameters **constant across a subject's sessions**
  (`drift={}`, `session_encoding_type=none`). Pure between-subject recovery.
- **Stage 2 (later):** within-subject **drift** across sessions (learn_rate ↑,
  |biasL| → 0, softmax_inv_temp ↑) + GRU **session conditioning** turned on.
- **Scale:** mice-comparable — `num_sessions_per_subject=40`, `num_trials=650`;
  `num_subjects` is swept {50, 100, 200, 300}.

## Two-part recovery scoring

1. **Fit quality — `likelihood_relative_to_groundtruth`** (headline).
   Model eval NL ÷ generating-policy eval NL, ceiling 1.0. Already logged by all
   three trainers (reads `avg_eval_likelihood_groundtruth` from the loader). A
   model that fits as well as the true policy scores ~1.0.
2. **Embedding recovery — R² / CCA** of the model's recovered subject embeddings
   against the true per-subject `(learn_rate, biasL, softmax_inv_temp)` centroids
   (analysis script, model-agnostic — reused across GRU and disRNN).
   Plus **baseline_rl** as the correct-model-class reference: it fits the *same*
   Q-learning structure per subject, so its `fitted_params_per_subject` vs truth
   is the achievable parametric-recovery ceiling.

## Held-out subjects: NOT used for the core question (ruling 2026-07-04)

This study asks **identifiability** ("does the embedding faithfully encode the
known true parameters?"), which is an **in-sample, representational-geometry**
question — scored on the *trained* subjects, whose embeddings the model actually
learns. It is deliberately NOT the data-scaling-law question (out-of-sample
*generalization* to new mice via zero/few-shot held-out eval).

Consequences:
- Core recovery metrics (relative likelihood + embedding-vs-truth CCA/R²) need
  **no held-out subjects**. `auto_heldout_finetune.enabled=false` in the GRU sweep
  and `heldout_refit.enabled=false` in baseline_rl are correct — the
  data-scaling-law few-shot/refit machinery is intentionally OFF.
- **Parked extension (revisit if in-sample recovery is clean):** *few-shot
  embedding recovery* — freeze the trained model, adapt ONLY a held-out subject's
  embedding from k sessions, and score inferred-embedding-vs-truth as a function of
  k. This reuses the data-scaling-law few-shot *mechanism* but reframes the *metric*
  from likelihood to parameter recovery (tests whether the embedding space is a
  usable coordinate system for cognition). Zero-shot on held-out subjects tests
  shared dynamics, not recovery, so it is not informative here.

## Variants index

| Variant | Question | Model | Grid | Status | W&B project | Group |
|---|---|---|---|---|---|---|
| `gru-stage1` | Between-subject embedding recovery vs #subjects, capacity, embed-size | GRU (multisubject) | subjects{50,100,200,300} × hidden{16,64,256} × embed{2,4,8} = 36 | scaffolded | `embedding_recovery` | `gru-stage1@<launch_id>` |
| `baseline-rl-stage1` | Correct-model-class recovery reference (likelihood ceiling + fitted-params-vs-truth) | baseline_rl / ForagerQLearning | subjects{50,100,200,300} = 4 | scaffolded | `embedding_recovery` | `baseline-rl-stage1@<launch_id>` |
| `gru-stage2` | Within-subject drift recovery + session conditioning | GRU (multisubject, SC on) | TBD | placeholder (built after Stage 1) | `embedding_recovery` | `gru-stage2@<launch_id>` |

disRNN replication of `gru-stage1` / `gru-stage2` is added as later variants once
the GRU results are in (recovery analysis is model-agnostic and reused as-is).

## Smoke test

A tiny plumbing run (4 subjects × 6 sessions × 80 trials, 20 steps) lives at
`code/hpc/sweeps/embedding_recovery_smoke.yaml` (+ `code/beaker/experiment_recovery_smoke.yaml`).
Validated end-to-end 2026-07-04 (HPC CPU, exit 0): loader → merged multisubject
dataset → `groundtruth_likelihood` + `likelihood_relative_to_groundtruth` logged,
`groundtruth_params.csv` written.

## Layout

```
studies/embedding-recovery/
  README.md                         # this file (Variants index)
  variants/
    gru-stage1/{sweep.yaml, experiment.yaml, notes.md}
    baseline-rl-stage1/{sweep.yaml, experiment.yaml, notes.md}
    gru-stage2/notes.md             # placeholder; sweep built after Stage 1
  analysis/                         # recovery-scoring scripts + reports (added at analysis time)
```

## How to launch (see beaker-launch / hpc-launch skills)

Beaker (GPU) from the Mac sandbox, or HPC SLURM CPU (4 CPUs/job when GPU is
scarce). The Beaker container pulls code at `WRAPPER_REF`/`DISPATCHER_REF`, so push
the branch and set those env refs — no image rebuild for code/config changes.
