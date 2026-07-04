# Variant gru-stage1 — static-subject GRU embedding recovery

**Goal.** Show that a multisubject GRU's learned subject embeddings recover the
*known* between-subject structure of the synthetic generator, and map how recovery
quality scales with #subjects, GRU capacity, and embedding size.

## Grid (36 points)

`data.num_subjects ∈ {50, 100, 200, 300}` ×
`model.architecture.hidden_size ∈ {16, 64, 256}` ×
`model.architecture.subject_embedding_size ∈ {2, 4, 8}`, `seed=42`.

All other knobs fixed: mice-comparable data (40 sessions/subject, 650 trials),
`session_encoding_type=none` (Stage 1: no session conditioning, subjects static
across sessions via `drift={}`), `n_steps=50000`, `lr=1e-3`, `batch_size=512`,
`eval_every_n=2`, `auto_heldout_finetune=false`.

## What differs from siblings

- vs `baseline-rl-stage1`: same synthetic data, but this is the neural model under
  test; baseline_rl is the correct-model-class recovery reference.
- vs `gru-stage2`: Stage 1 is static subjects, no session conditioning. Stage 2
  adds within-subject drift + `session_encoding_type` on.

## Expected

- `likelihood_relative_to_groundtruth` → toward 1.0 as #subjects and capacity grow
  (more data per shared dynamics, enough capacity to match the true policy).
- Embedding recovery R²/CCA (recovered subject embedding vs true
  `learn_rate, biasL, softmax_inv_temp`) → higher with more subjects; should
  saturate once embedding_size ≥ true dim (3) and capacity is sufficient.
- Over-large `subject_embedding_size` (8) should not hurt recovery if the extra
  dims stay unused (test via CCA, which is invariant to the embedding basis).

## Data determinism

Same `num_subjects`+`seed` regenerates byte-identical data across the
hidden/embed cells, so recovery differences within a #subjects column are
attributable to the model, not the data. Each run writes `groundtruth_params.csv`
(per-(subject,session) true params) alongside its outputs.

## Compute / tracking

- **Primary:** Beaker gcp-h100 (`experiment.yaml`), 1 GPU/agent; launch with
  `--count N` to spread the 36 points across agents.
- **Fallback (GPU scarce):** HPC SLURM CPU, 4 CPUs/job, many parallel array tasks.
- **WRAPPER_REF / DISPATCHER_REF:** `feat/embedding-recovery` (until merged up).
- **W&B:** project `embedding_recovery`, group `gru-stage1@<launch_id>`.

## Launch record

_(filled after launch: launch_id, SWEEP_ID, Beaker experiment id / SLURM job id,
then results.md once the group settles.)_
