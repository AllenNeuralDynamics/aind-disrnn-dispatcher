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
- `sweep_pack.yaml` / `experiment_pack.yaml` — GPU time-slicing benchmark (pack M
  agents on one GPU via the wrapper's `pack_gpu.sh`).

## Flow (dispatcher → wrapper hand-off)

```bash
WS=ai1/aind-dynamic-foraging-foundation-model

# 0. once per workspace: store the W&B key as a Beaker secret
beaker secret write han-wandb-api-key -w "$WS" "$WANDB_API_KEY"

# 1. create the sweep -> SWEEP_ID (e.g. AIND-disRNN/ai_hub_test/abc123)
wandb sweep code/beaker/sweep_mvp.yaml

# 2. put SWEEP_ID into experiment_mvp.yaml, then launch the Beaker job(s)
beaker experiment create -w "$WS" code/beaker/experiment_mvp.yaml
```

Monitor at `https://beaker.org/ex/<id>`; runs also appear in W&B `AIND-disRNN/ai_hub_test`.

## Clusters

**Rule: only use AI Hub _hub_ clusters** — those with `hub` in the name. The
non-hub clusters belong to other science units (our budget is
`ai1/aind-dynamic-foraging-foundation-model`); don't schedule on them. (Admin
guidance, 2026-06-17. Names are being renamed to a clearer convention ~mid-2026,
so re-check with `beaker cluster list ai1` or https://beaker.org/.)

Hub clusters — use these:

| Cluster | GPU | Notes |
|---|---|---|
| `ai1/octo-hub-aws-l40s` | L40s | our default — small models / debugging |
| `ai1/octo-hub-aws-l40s-dev` | L40s | dev |
| `ai1/octo-hub-aws-h200` | H200 | large training (AWS) |
| `ai1/octo-hub-onprem-h200` | H200 | on-prem |
| `ai1/octo.hub-gcp-h200` | H200 | GCP |
| `ai1/octo-hub-gcp-h100` | H100 | GCP |
| `ai1/aihub-dev-aws` | — | AI Hub dev |

Do **not** use (other units' allocations, not hub): `ai1/aipbd-aws-h200`,
`ai1/octo.ai-aws-p5en`, `ai1/octo.ai-aws-g6e`.

Pick one with free slots (`beaker cluster list ai1`); a job queues if none are
free. Slot caps (Allocated = non-preemptible, Unallocated = preemptible) are in
the AI Hub `getting-started/budgets.md`.

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
- **Time-slicing (many agents per GPU)** — the wrapper's `pack_gpu.sh` launches M
  `wandb agent` processes in one task with `XLA_PYTHON_CLIENT_MEM_FRACTION≈0.9/M`.
  Fills the headroom left by our host/eval-bound runs (100% util / ~30% power on L40s).
  Driven by `sweep_pack.yaml` + `experiment_pack.yaml`.

A meaningful sweep needs >1 combo, or extra agents have nothing to pull — hence the
multi-seed `sweep_scaling.yaml` / `sweep_pack.yaml`.

### Packing benchmark driver (M = 1, 4, 8)

`experiment_pack.yaml` has `<SWEEP_ID>` + `<PACK_M>` placeholders; loop over M,
creating a fresh sweep each time so each pack run gets a full queue:

```bash
WS=ai1/aind-dynamic-foraging-foundation-model
for M in 1 4 8; do
  SWEEP=$(wandb sweep code/beaker/sweep_pack.yaml 2>&1 | sed -n 's/.*wandb agent //p')
  tmp=$(mktemp)
  sed -e "s|<SWEEP_ID>|$SWEEP|" -e "s|<PACK_M>|$M|" code/beaker/experiment_pack.yaml > "$tmp"
  beaker experiment create -w "$WS" --name "disrnn-pack-m$M" "$tmp"
  rm -f "$tmp"
done
```

Compare **runs/GPU-hour + GPU power** across M; throughput should rise until GPU
compute or CPU (the host-bound limiter — bump `cpuCount` if so) saturates.
