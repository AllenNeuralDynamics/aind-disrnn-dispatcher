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

## Stage-1 finding (complete CPU grid: hidden∈{16,64} × embed∈{2,4} × N∈{50,100,200,300})
- **Fit likelihood is at ceiling (relative ≈ 0.98–1.00) for every cell** — a 2-D
  embedding fits the behaviour essentially perfectly. Likelihood alone cannot
  distinguish embedding sizes.
- **Recovery discriminates**: `embedding_size = 4` recovers all three parameters
  (mean R² ≈ 0.91–0.96, CCA r ≈ 0.99) at BOTH hidden sizes; `embedding_size = 2`
  is under-capacity (mean R² ≈ 0.42–0.50) — two dimensions cannot span three
  generating factors.
- **The neural embedding matches the correct-model reference.** baseline_rl
  (correctly-specified Q-learning, fitted per subject) recovers its own
  generating params at mean R² ≈ 0.91–0.97 — the achievable ceiling (imperfect
  only due to DE estimation noise, mostly in softmax_temp). GRU embed=4 recovery
  (0.91–0.96) essentially equals it: the learned subject embedding recovers the
  latent parameters as well as directly fitting the true model.
- **More hidden capacity does NOT rescue a too-small embedding — it can hurt.**
  At embed=2, going hidden 16→64 drops learn_rate recovery toward ~0 (the larger
  network fits behaviour without routing learn_rate cleanly through the 2-D
  bottleneck). The embedding size, not network size, is the identifiability knob.
- `softmax_inverse_temperature` is the hardest parameter for BOTH the GRU and the
  correct-model baseline (≈0.74–0.91), i.e. a property of the data
  (weak identifiability at the high-temperature end), not a GRU limitation.

Core methodological point: **likelihood alone would rate embed=2 and embed=4
equally; only embedding-vs-truth recovery reveals identifiability, and it needs
embedding_size ≥ number of latent generating factors.**

### Metric-scale gotcha (do not reintroduce)
baseline_rl logs **absolute** `eval_likelihood` (≈0.76 here), NOT the relative
ratio. To compare fit quality against GRU on one axis, divide by the
generating-policy likelihood: `baseline_relative = eval_likelihood /
groundtruth_likelihood` (per-N ground truth ≈ 50→0.7606, 100→0.7621, 200→0.7709,
300→0.7765; baseline relative then ≈ 0.9999, i.e. exactly at ceiling). Plotting
baseline-absolute against GRU-relative on a shared axis is a scale error that
makes the correct-model reference look worse than ceiling.
