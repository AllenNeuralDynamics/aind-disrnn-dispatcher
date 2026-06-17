# Beaker control plane (dispatcher)

The dispatcher is the **control plane** for Beaker runs — the same role it plays in
Code Ocean. It defines the W&B sweep and submits the Beaker experiment; the GPU
compute runs the **wrapper** image, built and maintained in
[`aind-disrnn-wrapper/beaker`](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/tree/ai_hub/beaker).

| | CO today | Beaker |
|---|---|---|
| Dispatcher (control) | composes Hydra config → job artifact | `wandb sweep` → **SWEEP_ID** → submit experiment |
| Wrapper (compute) | runs `run_hpc` | image runs `wandb agent` → `run_hpc` |

## Files
- `launch_beaker.py` *(in `code/`)* — the launcher invoked by a CO Reproducible Run:
  creates the sweep, saves a `/results` record, renders the SWEEP_ID, and submits.
- `sweep_mvp.yaml` — 1-point W&B sweep (smoke / default single run).
- `experiment_mvp.yaml` — single-GPU Beaker job spec.
- `sweep_scaling.yaml` — 8-combo (seed) sweep for testing parallelism.
- `experiment_scaling.yaml` — `replicas: 4`, the "array of jobs" across 4 GPUs.

## Flow (dispatcher → wrapper hand-off)

```bash
WS=ai1/aind-dynamic-foraging-foundation-model

# 0. once per workspace: store the W&B key as a Beaker secret
beaker secret write han-wandb-api-key -w "$WS" "$WANDB_API_KEY"

# 1. create the sweep -> SWEEP_ID (e.g. AIND-disRNN/beaker_mvp/abc123)
wandb sweep code/beaker/sweep_mvp.yaml

# 2. put SWEEP_ID into experiment_mvp.yaml, then launch the Beaker job(s)
beaker experiment create -w "$WS" code/beaker/experiment_mvp.yaml
```

Monitor at `https://beaker.org/ex/<id>`; runs also appear in W&B `AIND-disRNN/beaker_mvp`.

## Image & code version

The image is built on a Mac and pushed to Beaker — see the wrapper's
`beaker/README.md`. Code is pulled fresh at job startup, so **code edits need no
rebuild**; control the branch/commit via `WRAPPER_REF` / `DISPATCHER_REF` in
`experiment_mvp.yaml` (a branch name, or a SHA to pin a run).

## Scaling

Two ways to parallelize (both portable to HPC):

- **Array of jobs (across GPUs) — `replicas: N`.** N copies of the task, each its
  own GPU, all running `wandb agent` against the **same sweep**; W&B shards the
  combos. One-line spec change — see `experiment_scaling.yaml` (`replicas: 4`).
  Run it: `python code/launch_beaker.py --sweep code/beaker/sweep_scaling.yaml
  --experiment code/beaker/experiment_scaling.yaml`.
- **Time-slicing (many agents per GPU)** — launch M `wandb agent` processes in one
  task with `XLA_PYTHON_CLIENT_MEM_FRACTION≈1/M` (a `pack_gpu.sh`, not yet added).
  Layered on top of replicas to fill each big GPU with our tiny disRNN runs.

A meaningful sweep needs >1 combo, or extra agents have nothing to pull — hence
`sweep_scaling.yaml`'s 8 grid points.
