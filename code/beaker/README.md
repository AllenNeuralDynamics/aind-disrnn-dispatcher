# Beaker control plane (dispatcher)

The dispatcher is the **control plane** for Beaker runs — the same role it plays in
Code Ocean. It defines the W&B sweep and submits the Beaker experiment; the GPU
compute runs the **wrapper** image, built and maintained in
[`aind-disrnn-wrapper/beaker`](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/tree/ai_hub/beaker).

> **Two-repo layout.** This README is the **control plane** (sweeps, experiment
> specs, clusters, submission). The **compute / image plane** — building the image,
> the runtime code-pull, and how the job runs — lives in the wrapper:
> [`aind-disrnn-wrapper/beaker/README.md`](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/blob/ai_hub/beaker/README.md).
> Migration progress & benchmark results live there too:
> [Migration status](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/blob/ai_hub/beaker/README.md#migration-status)
> · [Benchmark figure](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/blob/ai_hub/beaker/README.md#benchmark-figure)
> · [Performance notes](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/blob/ai_hub/beaker/README.md#performance-notes-gpu-efficiency).

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

Hub clusters — use these (resources measured 2026-06-21; re-check with
`beaker cluster list ai1` / `beaker node get <id> --format json`):

| Cluster | GPU (mem) | Host RAM/node | Reaches DB? | Notes |
|---|---|---|---|---|
| `ai1/octo-hub-aws-l40s` | L40s (48 GB) | ~373 GiB (1 node, 4 slots) | ✅ AWS | default; fine for H128. 48 GB GPU OOMs a *wide* (hidden_size=256) full-cohort eval unless chunked |
| `ai1/octo-hub-aws-h200` | H200 (141 GB) | large | ✅ AWS | large training; often full (32/32) |
| `ai1/octo-hub-onprem-h200` | H200 (141 GB) | ~3.25 TiB | ✅ on-prem | large training; usually has free slots — best for wide H256 |
| `ai1/octo-hub-gcp-h100` | H100 (80 GB) | ~1.83 TiB | ❌ **cannot reach AWS S3 DB** | lots of free CPU/RAM, but DB reads fail (DNS / SSL-cert errors) — only for compute that doesn't touch the DB |
| `ai1/octo.hub-gcp-h200` | H200 (141 GB) | large | ❌ GCP (S3 unreliable) | |
| `ai1/octo-hub-aws-l40s-dev` | L40s (48 GB) | — | ✅ AWS | dev |
| `ai1/aihub-dev-aws` | T4 (16 GB) | 16 GiB | ✅ AWS | dev / tiny |

**DB-backed runs must use an AWS or on-prem cluster.** The trial/session
database is public AWS S3 (`s3://aind-scratch-data/aind-dynamic-foraging-cache`,
us-west-2); **GCP clusters cannot reliably read it** (intermittent
`Could not resolve hostname` / `SSL CA cert` `IOException`s mid-fetch). The DB
fetch itself is fast on AWS (~5 s for all ~12.5M trials; scales with CPU count).

Do **not** use (other units' allocations, not hub): `ai1/aipbd-aws-h200`,
`ai1/octo.ai-aws-p5en`, `ai1/octo.ai-aws-g6e`.

Pick one with free slots (`beaker cluster list ai1`); a job queues if none are
free. Slot caps (Allocated = non-preemptible, Unallocated = preemptible) are in
the AI Hub `getting-started/budgets.md`.

## Memory pitfalls (multisubject cohorts)

Large multisubject mice cohorts (`mice_snapshot_scaling`: ~600 subjects, ~18k
sessions, ~9.7M trials) hit three distinct memory walls. All are fixed in the
wrapper (branch `ai_hub_pck_integration`); notes here so they don't resurface.

1. **Slow load (host CPU), ~6–7 min.** `_build_multisubject_bundle` filtered
   `df[df["subject_id"] == sid]` once per subject over an object-dtype column —
   an O(subjects × trials) Python scan. Replaced with one `groupby("subject_id")`
   pass (and `create_disrnn_dataset` per-session `df.query()` → `groupby`, PR'd to
   `aind_disrnn_utils`). Load dropped to ~2 min. The DB fetch was never the
   bottleneck (~5 s); it's post-fetch dataset construction.

2. **Host-RAM OOM at eval.** The whole-cohort per-trial frame (`latent_*` ×
   hidden_size + logits/probs, millions of rows) was built unconditionally just
   to plot a few subjects → 100–300 GB RSS, OOM-killed (SIGKILL, no traceback) on
   the 373 GiB L40s node — worse with several `replicas` sharing one node. Fixed:
   build the per-trial frame only for the subjects actually plotted; training
   metrics (likelihood) read the `yhat` tensors directly and never need it.

3. **GPU OOM at eval.** `eval_network` JIT-runs the model over *all* sessions at
   once; for hidden_size=256 that allocated ~39.5 GB and OOM'd the 48 GB L40s GPU
   (`RESOURCE_EXHAUSTED`). Fixed: chunk the forward pass over the session axis
   (`GruTrainer._eval_max_episodes`). hidden_size=128 fits an L40s as-is; 256 fits
   with chunking, or run on an H200 (141 GB).

Operational guardrail: the L40s node is **one machine, 4 GPUs, ~373 GiB shared**.
Cap co-location with `resources.memory` + `replicas` so concurrent runs don't
exceed node RAM (≈ `373 GiB / replicas`). Wide (H256) runs: prefer an H200, or
L40s with the eval chunking above and `replicas`≤2.

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

> **Result:** packing this workload plateaus at ~1.15× (no-MPS time-slicing
> serializes the low-occupancy kernels) — batch size is the real lever. Full
> findings in the wrapper's
> [Performance notes](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/blob/ai_hub/beaker/README.md#performance-notes-gpu-efficiency)
> and [benchmark figure](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/blob/ai_hub/beaker/README.md#benchmark-figure).
