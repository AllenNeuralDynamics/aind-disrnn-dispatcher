# Study: 04-gru-vs-disrnn-embedding-recovery

> **formerly:** `embedding-recovery` (renamed 2026-07-11 to the `NN-{model}-{purpose}`
> convention; accession `04`, cross-model since the study now spans GRU and disRNN).
> The W&B **project** is still `embedding_recovery` and the `meta.study` stamp on
> already-logged runs is unchanged; historical launch records keep the path used at
> launch time.

**Scientific question.** When a data-driven sequence model (GRU, then disRNN) is
trained on multisubject foraging data, do its learned **subject embeddings** (and
its **session conditioning**) actually recover the *known* generative structure of
the animals — or are they arbitrary codes that happen to fit?

We answer this on **synthetic data with ground truth**: each synthetic "subject"
is a Q-learning-family agent occupying a distinct subregion of parameter space, so
we know the true per-(subject, session) parameters and the true generating policy.

## Verdict

**The data-driven embedding recovers generative structure exactly where a
correctly-specified baseline stops being sufficient.** A correct-model baseline
matches the GRU on stationary (S1) and interpolable (S2) data, then flips: it drops
to 0.94 relative likelihood under extrapolation (S2b) and to 47%/70% model-selection
accuracy under mixed structure (S3/S4a), while the GRU embedding recovers the true
parameters, model variant, and model family at **97.5–100%**. Embedding dimension —
not network width — is the identifiability knob. The interpretable, bottlenecked
**disRNN** replicates the Stage-4a family recovery at **0.95–0.98** (with session
conditioning; 0.75–0.90 without) for a modest **~4–6 point** likelihood cost — so
interpretability is nearly free for recovery. The one negative result is
informative: in Stage-4b (sparse per-session family mixtures) recovery lives at the
**subject** level, and session conditioning adds nothing, because Dirichlet(0.5)
subjects are concentrated enough that a session's family is fixed by subject
identity.

Full results: [`analysis/reports/INDEX.md`](analysis/reports/INDEX.md)
(r1 = GRU ladder, r2 = disRNN replication). Regenerate everything with
`make -C studies/04-gru-vs-disrnn-embedding-recovery`.

## Ground-truth generative model

Data is produced on-the-fly by `HierarchicalCognitiveAgents`
(`aind-disrnn-wrapper/code/data_loaders/hierarchical_synthetic.py`,
Hydra `data=synthetic_hierarchical`). It is **deterministic** given a seed
(instance-RNG task + agent, non-overlapping seed hierarchy), so no frozen dataset
is needed — every run regenerates byte-identical data and writes a
per-(subject, session) ground-truth parameter table (`groundtruth_params.csv`).

- **Base agent (Stages 1–2b):** `ForagerQLearning` — `number_of_learning_rate=1`,
  `number_of_forget_rate=0`, `choice_kernel=none`, `action_selection=softmax`.
- **Recoverable generating parameters (the dim-3 ground-truth embedding):**
  `learn_rate` ~ U[0.1, 0.9], `biasL` ~ U[-1.5, 1.5],
  `softmax_inverse_temperature` ~ U[2, 15]. Each subject = one centroid.
- **Drift (Stage-2 onward):** within-subject drift across sessions; Stage-2b adds
  strong + non-monotonic drift with a tail held-out session split.
- **Mixtures (Stages 3–4):** Stage-3 mixes QL variants (Bari/Hattori/RW), Stage-4a
  mixes model *families* (QL/CompareToThreshold/LossCounting) one per subject,
  Stage-4b switches family per session from a sparse Dirichlet(0.5) subject mixture.
- **Scale:** mice-comparable — `num_sessions_per_subject=40`, `num_trials=650`;
  `num_subjects` swept {50, 100, 200, 300} (Stages 1–2), fixed at 200 (Stages 2b–4).

## Two-part recovery scoring

1. **Fit quality — `likelihood_relative_to_groundtruth`** (headline).
   Model eval NL ÷ generating-policy eval NL, ceiling 1.0. Logged by all three
   trainers. A model that fits as well as the true policy scores ~1.0.
2. **Embedding recovery** — R²/CCA of the recovered subject embeddings vs the true
   per-subject parameters (Stages 1–2), and classification of model variant/family
   from the embedding (Stages 3–4). Model-agnostic, reused across GRU and disRNN.
   **`baseline_rl`** is the correct-model-class **reference** (not a competitor):
   it fits the same structure per subject, so its `fitted_params_per_subject` vs
   truth is the achievable parametric-recovery ceiling, and its per-subject
   model-selection accuracy is the achievable structure-recovery ceiling.

## Held-out subjects: NOT used for the core question

This study asks **identifiability** (in-sample, representational-geometry: does the
embedding faithfully encode the known true parameters?), scored on the *trained*
subjects. It is deliberately NOT the data-scaling-law question (out-of-sample
generalization to new mice). So `auto_heldout_finetune.enabled=false` (GRU) and
`heldout_refit.enabled=false` (baseline_rl) are correct — the few-shot/refit
machinery is intentionally OFF. (Stage-2b onward adds a *tail session* held-out
split — a different mechanism, evaluated via the normal eval-likelihood path.)

## Variants index

19 variants across 6 ladder stages. W&B project `embedding_recovery`
(entity `AIND-disRNN`); group = `<variant>@<launch_id>`. Status as of 2026-07-11.

| Variant | Stage | Model | Generator axis | Grid | W&B sweep | Status | Headline result |
|---|---|---|---|---|---|---|---|
| `gru-stage1` | 1 | GRU | static (no drift) | N{50,100,200,300}×{none,scalar}×hid{16,64} | `mfuaz3ki` | done | subject param R² 0.91–0.96 @D4 |
| `baseline-rl-stage1` | 1 | baseline QL | static reference | N-grid | `s02vaygf` | done | structure-matched reference (ceiling) |
| `gru-stage2` | 2 | GRU | mild monotonic drift | N{50,100,200,300}×{none,scalar} | `l0gcdrqh` | done | subj R² 0.96 (scalar); sess-frac R² up to 0.94 |
| `baseline-rl-stage2` | 2 | baseline QL | mild-drift reference | N-grid | `n53a3qc6` | done | rel-LL 0.993 (at ceiling) |
| `gru-stage2b` | 2b | GRU | strong + non-monotonic drift, tail held-out | {none,scalar}, N=200 | `c838tjrw` | done | **baseline flip**: GRU >0.987 vs baseline 0.939 |
| `baseline-rl-stage2b` | 2b | baseline QL | strong drift, extrapolation | N=200 | `ykjk89o5` | done | rel-LL 0.939 (drops under extrapolation) |
| `gru-stage3` | 3 | GRU | QL-variant mixture (Bari/Hattori/RW) | embed{4,8,16}×{none,scalar} | `ychlajgl` | done | preset classification 97.5–99.5% |
| `baseline-bari-stage3` | 3 | baseline Bari | fixed-model reference | N=200 | `7hryk8zk` | done | model-selection 47% (3 baselines) |
| `baseline-hattori-stage3` | 3 | baseline Hattori | fixed-model reference | N=200 | `awot1o8i` | done | (part of 47% model-selection) |
| `baseline-ctt-stage3` | 3 | baseline CTT | fixed-model reference | N=200 | `ytjas9yl` | done | (part of 47% model-selection) |
| `gru-stage4a` | 4a | GRU | family mixture (QL/CTT/LC) | embed{4,8,16}×{none,scalar} | `93pci64o` | done | family decoding 100% |
| `baseline-bari-stage4a` | 4a | baseline Bari | fixed-family reference | N=200 | `uh8166n4` | done | model-selection 70% (3 baselines) |
| `baseline-ctt-stage4a` | 4a | baseline CTT | fixed-family reference | N=200 | `si2fnnv3` | done | (part of 70% model-selection) |
| `baseline-losscounting-stage4a` | 4a | baseline LC | fixed-family reference | N=200 | `b493xnry` | done | (part of 70% model-selection) |
| `gru-stage4b` | 4b | GRU | per-session family switching | embed{4,8,16}×{none,scalar} | `nptb5bam` | done (5/6)¹ | mix-weight R² 0.55 @D16; per-session family 0.62 |
| `baseline-bari-stage4b` | 4b | baseline Bari | per-session-switch reference | N=200 | `0w5f39p7` | done | reference |
| `baseline-ctt-stage4b` | 4b | baseline CTT | per-session-switch reference | N=200 | `nrehx3ox` | done | eval-LL 0.706 |
| `baseline-losscounting-stage4b` | 4b | baseline LC | per-session-switch reference | N=200 | `b3m2un3h` | done | reference |
| `disrnn-stage4a` | 4a | **disRNN** | family mixture (replication) | embed{4,8,16}×{none,scalar} | `w2628h00` | done | family decode 0.95–0.98 scalar / 0.75–0.90 none; rel-LL ~0.94 |

¹ `gru-stage4b` none-D8 crashed at ~90% (non-blocking; the other 5 cells + all 3
baselines completed). disRNN Stage-4a was trained on `octo-hub-onprem-h200` after
AWS p5en preemption churned the initial cells — see
`variants/disrnn-stage4a/launch_record/results.md`.

## Layout

```
studies/04-gru-vs-disrnn-embedding-recovery/
  README.md              # this file (question, Verdict, Variants index)
  FUTURE_DIRECTIONS.md
  Makefile               # `make` regenerates both reports from committed grids
  environment.lock       # pins the wrapper commit that trained + scored the grids
  CHANGELOG.md
  .gitignore             # caches ignored; curated JSON/CSV/PNG committed
  analysis/
    recovery_report.py   # SINGLE producer: grids -> summary.{json,csv} + 3 figs + report blocks
    wandb_keys.py        # comment-only W&B summary-key contract
    ladder_results.csv, stage4b_recovery_grid.csv, disrnn_stage4a_recovery_grid.csv  # committed inputs
    recovery_summary.{json,csv}, fig_*.png                                            # committed outputs
    figures/             # 10 per-stage recovery + embedding-space figures, each embedded in r1
    reports/{INDEX.md, r1-gru-ladder.md, r2-disrnn-replication.md}  # results; per-stage figs embedded inline in r1
    provenance/wandb_project_url.txt
  variants/<variant>/{sweep.yaml, notes.md, launch_record/{results.md, _sweeps/…}}
```

## How to launch (see hpc-launch / beaker-launch skills)

HPC SLURM CPU (`launch_hpc.py`, 8 CPUs/job) for the GRU + baseline ladder; Beaker
GPU (onprem-h200) for the disRNN. The Beaker container pulls code at
`WRAPPER_REF`/`DISPATCHER_REF`, so push the branch and set those env refs — no image
rebuild for code/config changes.
