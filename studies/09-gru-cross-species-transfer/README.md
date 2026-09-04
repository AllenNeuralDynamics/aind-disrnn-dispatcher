# Study 09: GRU cross-species and cross-dataset transfer

Issues: dispatcher [#32](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/32),
[#126](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/126),
and [#127](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/127);
wrapper [#91](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/91)
and [#92](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/92).

This starter suite tests whether the GRU core learned from AIND dynamic
foraging transfers to external two-arm bandit behavior. The priority order is:

1. Grossman, Bari & Cohen: 48 mice, blockwise independent reward probabilities.
2. Chen et al.: 32 mice, eight restless-bandit sessions.
3. Zid et al. Experiment 1: 258 humans, schedule-matched to Chen.

## Build the suite

The command below downloads pinned public releases, verifies their repository
checksums, converts them to the wrapper's canonical trial table, writes an
explicit split manifest, and fails if subject/session/trial counts differ from
the audited release.

    make suite

Set CACHE_ROOT to place the uncommitted raw and canonical data elsewhere. After
the first successful download, use make rebuild to work entirely from the
checksum-verified local source files.

Generated files are:

- canonical/{grossman,chen,zid}.parquet: one row per valid decision trial;
- canonical/{grossman,chen,zid}.split.json: the immutable adaptation/test split;
- canonical/{grossman,chen,zid}.audit.json: provenance, checksums, and counts.

The six required columns are subject_id, ses_idx, trial, animal_response,
rewarded, and earned_reward. Choices and rewards are binary. The adapters retain
source trial numbers, reward probabilities, and available cohort metadata as
additional columns.

## Cohort-specific decisions

Grossman uses only the dynamicForaging/behavior branch. This is the curated
48-mouse behavior cohort and avoids duplicate sessions also present under neural
or chemogenetics analyses. CSminus no-go trials and CSplus trials without a
left/right decision are excluded. Left is encoded 0 and right 1.

Chen retains all 256 source files. Source choices 1/2 become canonical choices
0/1. All eight sessions remain separate, including short sessions.

Zid uses the official Experiment 1 MATLAB file rather than the equivalent
Python pickle, avoiding executable deserialization of downloaded data. The
first 25 fixed-schedule practice trials are excluded, leaving the 300 main
trials for each participant.

## Frozen split contract

For Grossman and Chen, sessions are ordered chronologically within subject. The
first, third, fifth, and later odd-positioned sessions are adaptation data; the
second, fourth, sixth, and later even-positioned sessions are test data. This is
the current GRU study convention (eval_every_n=2). Therefore a K-session
few-shot run takes the first K sessions *within that odd-positioned adaptation
sequence*, not the first K sessions before the odd/even split.

Zid has one main session per person, so inventing pseudo-sessions would reset
the recurrent and Q-learning states at an artificial boundary. Instead, the
manifest assigns trials 0-149 to an adaptation prefix and 150-299 to a test
suffix. Evaluation runs the prefix to establish the state at the boundary, then
scores the suffix without changing fitted parameters. Observed choices and
outcomes still update the recurrent or Q state online during the suffix; only
the learned parameters remain frozen.

## Benchmark matrix

Each external dataset is held out from foundation-model training. Evaluate:

- zero-shot: frozen GRU core and the prespecified new-subject embedding
  initialization;
- few-shot: frozen GRU core, adapt only the new-subject embedding on K
  adaptation sessions (or a declared prefix budget);
- half-data shot: the full adaptation half, again changing only the embedding;
- subject-level Q-learning: fit parameters on the identical adaptation
  observations and score the identical test observations.

The primary comparison is mean per-trial log likelihood on the test partition,
paired by subject. Report normalized likelihood, Brier score, accuracy, and
calibration as descriptive secondary metrics. Fit preprocessing, hyperparameters,
random seeds, split manifests, source checksums, and model checkpoints are part
of the run provenance.
