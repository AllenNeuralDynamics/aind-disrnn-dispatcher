# Variant baseline-rl-stage1 — correct-model-class recovery reference

**Goal.** Provide the achievable ceilings that the GRU's recovery is measured
against, by fitting the *true* model class (independent per-subject Q-learning)
to the same synthetic data.

## Two reference roles

1. **Likelihood ceiling.** With the fitter structure-matched to the generator,
   `likelihood_relative_to_groundtruth` should sit at ~1.0 — the best any model
   can do on this data. The GRU's headline score is read relative to this.
2. **Parametric-recovery ceiling.** `fitted_params_per_subject`
   (`baseline_rl_trainer.py`) gives per-subject fitted
   `(learn_rate, biasL, softmax_inverse_temperature)`. Scored against the true
   centroids, this is the best achievable direct parameter recovery — the
   reference band for the GRU embedding-recovery R²/CCA.

## Grid (4 points)

`data.num_subjects ∈ {50, 100, 200, 300}`, `seed=42` — one per #subjects column,
matching gru-stage1 exactly (same synthetic data).

## CRITICAL: structure match

`config/model/baseline_rl.yaml` defaults to a RICHER fitter than the generator
(`number_of_forget_rate=1`, `choice_kernel=one_step`). The sweep OVERRIDES to
`number_of_forget_rate=0`, `choice_kernel=none` (`number_of_learning_rate=1`,
`action_selection=softmax`) so the fitter is **correctly specified** and recovered
params map 1:1 to the 3 generating params. Without this override the model still
fits well, but recovered params live in a different (over-parameterized) space and
can't be compared to ground truth.

## Compute

`baseline_rl` is **CPU-bound** (scipy differential-evolution, `polish=true`,
`multisubject_subject_workers=6`) — run on **HPC SLURM CPU**, not a GPU. At
mice scale (300 subjects) DE fitting is the long pole; use several CPUs/job and
parallelize the 4 points across array tasks.

Launch (HPC, from a feat/embedding-recovery checkout/worktree):

```
python code/launch_hpc.py \
  --sweep-yaml studies/embedding-recovery/variants/baseline-rl-stage1/sweep.yaml \
  --mode cpu --wrapper-root <wrapper> \
  --label baseline-rl-stage1 --note "correct-model-class recovery reference"
```

`experiment.yaml` (Beaker) is provided only for parity/CPU-on-Beaker if ever
needed; the HPC CPU route is the intended path.

## W&B / tracking

Project `embedding_recovery`, group `baseline-rl-stage1@<launch_id>`.

## Launch record

_(filled after launch.)_
