# Embedding-recovery — analysis

Reproducible, model-agnostic scoring for the Stage-1 (and later Stage-2)
subject/session embedding-recovery runs. Works for GRU, disRNN, and (via the
absolute-likelihood key) baseline_rl.

## Files
- `recovery_scoring.py` — the scorer library. Two-part recovery score:
  1. **Fit**: `likelihood_relative_to_groundtruth` (model NL / generating-policy
     NL, ceiling 1.0). GRU/disRNN log it directly; baseline_rl logs absolute
     `eval_likelihood`, so its relative score = `eval_likelihood /
     groundtruth_likelihood`.
  2. **Recovery**: how well the learned subject embedding encodes the true
     generating parameters (`biasL`, `learn_rate`, `softmax_inverse_temperature`):
     - `ridge_r2` — 5-fold cross-validated R² of predicting each true param from
       the embedding (embedding → param). Interpretable and robust to
       `embedding_size ≠ 3`.
     - `cca_scores` — CCA canonical correlations between embedding and params.
- `run_recovery_analysis.py` — end-to-end runner: enumerates finished W&B sweep
  runs, loads each `subject_embeddings.pkl`, scores it against the ground-truth
  master table, and writes a tidy `recovery_scores.csv`.

## Ground truth
Each training run writes `groundtruth_params.csv` (one row per (subject,
session)) plus `groundtruth_summary.json`. In **Stage 1** the per-subject params
are **static across sessions** and **deterministic in `subject_idx` alone**
(centroid seed = `base + subject_idx × subject_seed_stride`, independent of
`num_subjects`) — verified byte-identical across the 200- and 300-subject tables.
So the largest run's table is a superset that serves every run; pass it as
`--gt-master`. (In Stage 2, params drift within subject — recovery is then scored
per (subject, session) and this static-collapse no longer applies.)

## Reproduce
```bash
# env: numpy pandas scipy scikit-learn matplotlib seaborn
python run_recovery_analysis.py \
    --entity AIND-disRNN --project embedding_recovery \
    --sweeps <gru_sweep_id> [<gpu_sweep_id> ...] \
    --embeddings-root <dir with <run_id>/subject_embeddings.pkl> \
    --gt-master <path/to/groundtruth_params.csv> \
    --out-dir ./analysis_out
```

On Allen HPC / Code Ocean, `wandb.Api()` is unrestricted; from a
network-restricted sandbox use the raw GraphQL POST already wired into
`fetch_sweep_runs()` (`https://api.wandb.ai/graphql`, `auth=("api",
WANDB_API_KEY)`).

## Stage-1 finding (preliminary, hidden=16 + first hidden=64 cells)
- **Fit likelihood is at ceiling (0.98–0.999) for every embedding size** — a 2-D
  embedding fits the behaviour essentially perfectly.
- **Recovery discriminates**: `embedding_size = 4` recovers all three parameters
  (R² ≈ 0.93–0.99, CCA r ≈ 0.995); `embedding_size = 2` is under-capacity
  (R²_mean ≈ 0.42) — two dimensions cannot span three generating factors.
- `biasL` is the most sample- and capacity-hungry parameter (R² 0.80 → 0.95 as
  N grows 50 → 200 at hidden=16; 0.80 → 0.93 going hidden=16 → 64 at N=50).

This is the core methodological point: **likelihood alone would rate both
embedding sizes equally; only embedding-vs-truth recovery reveals identifiability.**
