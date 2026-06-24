# Variant rl-baseline-simple

**Goal:** add a simple classical RL reference for the mouse data-scaling study.
This is the same baseline family as Po-Chen's prior train10 work: fit an
interpretable dynamic-foraging RL model independently per subject.

## Interpretation

- **Model:** `baseline_rl` / `ForagerQLearning`, `number_of_learning_rate=1`,
  `number_of_forget_rate=1`, `choice_kernel=one_step`, `action_selection=softmax`.
- **Fit:** one RL parameter vector per subject via differential evolution.
- **Held-out scoring:** for each reserved held-out mouse, fit a fresh RL agent on
  that mouse's held-out train sessions and score that same mouse's held-out eval
  sessions (`eval_every_n=2`, every second ordered session is eval).
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
for this reference that the default launch should not spend three full held-out
refits; add seeds `[1, 2]` only as an optional stability check if the first run
looks suspicious. It fixes `data.subject_ratio=0.016`
only to keep the required training-subject fit small; the reported reference is
`heldout/eval_likelihood`, produced by held-out re-fit on the fixed reserved
held-out cohort.

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
  --label simple-rl \
  --note "independent per-subject RL reference: fit fresh RL params per held-out mouse train sessions; score held-out eval sessions; horizontal baseline, not D-scaling"
```

The launcher creates the W&B sweep in
`https://wandb.ai/AIND-disRNN/mice_data_scaling`, injects dispatcher/wrapper git
lineage, stamps `meta.{study,variant,launch_id,config_hash,label,note}`, sets the
W&B group to `<variant>@<launch_id>`, and submits CPU-only Slurm agents.

## Status

Ready to launch; no runs recorded yet.
