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
- `launch_beaker_resumable.py` *(in `code/`)* — Option-1 **resumable** launcher:
  expands a `method: grid` sweep into one autoResume Beaker task per grid point
  (see "Resumable runs" below).

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

3. **GPU OOM (hidden_size=256).** The model runs a forward over *all* ~18k
   sessions at once; for hidden_size=256 that allocates ~37-40 GB on the GPU and
   `RESOURCE_EXHAUSTED`s anything smaller. The binding site is **training**
   (`gru_trainer.fit` -> `train_network_with_session_regularization`), confirmed
   on both an unpacked 48 GB L40s and a 4-way-packed share. The **eval** forward
   has the same full-cohort shape and is chunked over the session axis
   (`GruTrainer._eval_max_episodes`) — a real memory win, but it does NOT remove
   the training allocation. So: **hidden_size=256 needs a big GPU (H200, 141 GB)**;
   it does not fit a 48 GB L40s or a packed M=4 share. hidden_size=128 (~half)
   fits an L40s and packs fine. Chunking the *training* forward would be a further
   change (not done).

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

## Resumable runs (preemption recovery)

Packed `wandb agent` sweeps (above) are *not* preemption-resilient: when a
preemptible agent is killed mid-trial, Beaker restarts the container, the new
agent pulls a **fresh** sweep trial, and the interrupted trial's partial
progress is thrown away (grid coverage is preserved — wandb re-dispatches the
cell — but the half-trained model is lost). For long runs (wide hidden sizes,
high step counts) that wasted compute is the cost worth eliminating.

`launch_beaker_resumable.py` runs a grid the resumable way instead:

```bash
WS=ai1/aind-dynamic-foraging-foundation-model
python code/launch_beaker_resumable.py \
  --sweep code/beaker/sweep_gru_scaling.yaml \
  --experiment code/beaker/experiment_scaling.yaml \
  --workspace "$WS"
# add --no-submit to render + inspect the spec without launching
```

It expands the grid into one **self-contained Beaker task per grid point**
(running `run_hpc` with that point's Hydra overrides baked in — no sweep
controller), each `preemptible: true` / `priority: low`. Three things make a
preempted task resume from its last checkpoint instead of restarting:

1. **Beaker autoResume** — set `preemptible: true` (with `priority: low`) and the
   server applies `autoResume: true` automatically (confirm with
   `beaker experiment spec <id>`). Do **not** set `autoResume` explicitly in the
   spec: the v2 schema rejects `preemptible` and `autoResume` together
   (`Error: preemptible cannot be set with min_runtime or auto_resume`). The
   restart re-runs the *same* command and re-attaches the *same* `/results`
   dataset — verified live: a `beaker job preempt`ed run resumed from
   `/results/run/outputs/checkpoints/step_300/train_state.pkl`, skipped warmup,
   and continued with stable likelihood.
2. **Stable output dir** — each task sets `DISRNN_RESUMABLE_OUTPUT_DIR=/results/run`
   so `run_hpc` anchors outputs at a fixed path (not the per-run W&B dir), so the
   restart re-finds `checkpoints/step_<N>/train_state.pkl`.
3. **Full-state checkpoints** — the trainer writes params + optimizer + PRNG key
   + step each checkpoint and, on startup, resumes from the highest one (the
   wrapper's `model_trainers/checkpoint_resume.py`; gated by
   `training.auto_resume`, default true). Requires `training.checkpoint_every_n_steps > 0`.

W&B continuity across the restart is preserved by a deterministic, per-grid-point
`WANDB_RUN_ID` + `WANDB_RESUME=allow`, with all points grouped under
`WANDB_RUN_GROUP`.

Only `method: grid` sweeps are supported — the trial set must be enumerable up
front, since there is no sweep controller to ask for the next point.
