# Figure gallery — model recovery & parameter recovery, Stages 1→4

Every ladder stage has a recovery figure and, from Stage 3 on, an embedding-space
visualization (subject embedding → PCA 2D, colored by true family/type and by key
parameters). This is the canonical deliverable set; the summary figures the
producer regenerates (`fig_ladder.png`, `fig_disrnn_stage4a.png`, `fig_stage4b.png`)
sit alongside these under `analysis/`.

Regenerable summary figures (produced by `recovery_report.py`):
- `fig_ladder.png` — the baseline flip across the whole ladder (S1→S4b).
- `fig_disrnn_stage4a.png` — disRNN family decoding, likelihood cost, **embedding-space PCA**.
- `fig_stage4b.png` — Stage-4b mixture-weight recovery vs embedding size.

Per-stage recovery figures (in `analysis/figures/`, produced by the per-stage
scripts `run_recovery_analysis.py` / `make_recovery_figures.py` /
`stage2_recovery.py` / `stage2_session_traj.py`):

## Stage 0 — setup
- `recovery_ground_truth_schematic.png` — what must be recovered: per-subject param
  subregions + within-subject drift; the subject-embedding table + session-MLP targets.

## Stage 1 — static subjects (parameter recovery)
- `stage1_recovery_vs_baseline.png` — (a) mean recovery R² vs #subjects, GRU vs
  correct-model baseline; (b) fit quality all at ceiling; (c) per-parameter R².
  **Embedding size, not net width, is the identifiability knob** (h16≈h64).

## Stage 2 — mild within-subject drift
- `stage2_recovery.png` — subject-level parameter recovery R² vs #subjects.
- `stage2_likelihood_comparison.png` — all models near ceiling; likelihood can't
  separate them (motivates the recovery axis).
- `stage2_session_trajectory.png` — **only the session-conditioning MLP encodes
  drift position** (subject-only delta-zeroed = R² 0.00 by construction); (c) each
  subject traces a drift path in embedding space.

## Stage 2b — strong + non-monotonic drift, held-out tail
- `stage2b_likelihood_flip.png` — **the baseline flip**: static Q-learning collapses
  (0.939) under extrapolation while both GRUs stay >0.987; (b) where model
  separation now lives (gap breakdown).
- `stage2b_session_trajectory.png` — stronger drift → session delta adds more;
  non-monotonic makes position harder (0.94→0.47) but delta≠0; (c) drift paths.

## Stage 3 — mixture of Q-learning variants (Bari/Hattori/RW)
- `stage3_recovery_combined.png` — 6-panel: **embedding-space PCA** (a colored by
  true type, b shape=type/color=biasL, c shape=type/color=learn_rate),
  (d) type decoded from embedding 97.5–99.5%, (e) confusion, (f) within-family
  parameter recovery. **Model TYPE → which cluster; PARAMETERS → position within.**
- `stage3_embedding_space.png` — the standalone 3-panel embedding-space view.
- `stage3_baseline_vs_gru_confusion.png` — GRU embedding decode vs baseline
  model-selection (47%).

## Stage 4a — mixture of model families (QL / CompareToThreshold / LossCounting)
- `stage4a_recovery_combined.png` — (a,b) **embedding-space PCA** separating the
  three families; (c) GRU embedding decode 100%; (d) fixed-baseline model selection 70%.
- `disrnn_stage4a_replication.png` — disRNN replication: family decode, likelihood
  cost, **embedding-space PCA** (regenerated as `fig_disrnn_stage4a.png`).

## Stage 4b — per-session family switching
- `stage4b_recovery.png` — (a) mixture-weight recovery vs embedding size;
  (b) subject-vs-session dissociation null; (c) per-session family confusion.
