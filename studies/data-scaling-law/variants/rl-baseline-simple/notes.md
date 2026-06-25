# Variant rl-baseline-simple

**Goal:** add a simple classical RL reference for the mouse data-scaling study.
This is the same baseline family as Po-Chen's prior train10 work: fit an
interpretable dynamic-foraging RL model independently per subject.

## Interpretation

- **Model:** `baseline_rl` / `ForagerQLearning`, `number_of_learning_rate=1`,
  `number_of_forget_rate=1`, `choice_kernel=one_step`, `action_selection=softmax`.
- **Fit/scoring set:** use the normal mice config but set
  `model.heldout_refit.skip_train_fit=true`; no training-subject fit is run.
- **Held-out scoring:** for each reserved held-out mouse, fit a fresh RL agent on
  that mouse's train sessions and score that same mouse's eval sessions
  (`eval_every_n=2`, every second ordered session is eval). In W&B this reports
  the aggregate as `heldout/eval_likelihood`, matching the GRU namespace.
- **What it tests:** whether the GRU beats a stable per-mouse cognitive model on
  the same held-out scoring sessions.
- **What it does not test:** whether more training mice improve a population
  model. There is no cross-mouse sharing, no subject/session embedding, and no
  D-scaling mechanism. Treat the result as a horizontal reference band, not a
  scaling curve.

The later fair population-model RL baseline is hierarchical/Bayesian RL: learn a
population prior from D training mice, adapt/posterior-fit each held-out mouse
from its train sessions, and score its eval sessions.

## Sweep

`sweep.yaml` runs one optimizer seed. Differential evolution is stable enough
for this reference that the default launch should not spend three full fits; add
seeds `[1, 2]` only as an optional stability check if the first run looks
suspicious. It uses `model.heldout_refit.skip_train_fit=true`, so the main
training-subject fit is skipped and only the fixed reserved held-out cohort is
fit/scored under `heldout/*`.

Parallelism is CPU-only:

- one Slurm task for the one default seed,
- `model.multisubject_subject_workers=112`,
- `model.DE_kwargs.workers=1` to avoid nested process pools.

The `aind` partition has CPU-only nodes up to 112 cores. Nodes above that
(288-core H200 nodes) are GPU nodes, and requesting multiple CPU nodes would not
help this Python process pool because it does not distribute one fit across
nodes. So the practical full-node CPU request is one 112-core CPU node.

## Launch

Run from the dispatcher repo on Allen HPC, using the `disrnn-cpu` control-plane
environment:

```bash
conda activate disrnn-cpu
python code/launch_hpc.py \
  --sweep-yaml studies/data-scaling-law/variants/rl-baseline-simple/sweep.yaml \
  --mode cpu \
  --sbatch-extra="--array=0-0 --nodes=1 --cpus-per-task=112 --mem=400G --time=24:00:00 --constraint=cpu" \
  --agent-count 1 \
  --label simple-rl-bari-l1f1-ck1 \
  --note "Bari L1F1_CK1 independent per-subject RL reference: directly fit reserved held-out mice train sessions; score held-out eval sessions; horizontal baseline, not D-scaling"
```

The launcher creates the W&B sweep in
`https://wandb.ai/AIND-disRNN/mice_data_scaling`, injects dispatcher/wrapper git
lineage, stamps `meta.{study,variant,launch_id,config_hash,label,note}`, sets the
W&B group to `<variant>@<launch_id>`, and submits CPU-only Slurm agents.

## Status

Done (2026-06-24). Single run on Allen HPC (CPU, 112 cores, ~24h DE fit).

- Launch: `rl-baseline-simple@20260624-171829`
- Run: [`cdq292n5`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/cdq292n5)
- Coverage: 149 held-out mice (1.01M eval trials)
- Aggregates: trial-weighted pooled LL **0.7143**; per-subject mean **0.7211** ± 0.0052 SE; median 0.7305
- GRU beats per-mouse RL by **+0.0136** at v2 D=614 (100% of mice, Wilcoxon p=3e-26); even v2 D=10 wins on 97% of mice. See [`analysis/rl_baseline_verdict.md`](../../analysis/rl_baseline_verdict.md) and Result 8 in [`analysis/FINAL_REPORT.md`](../../analysis/FINAL_REPORT.md).
