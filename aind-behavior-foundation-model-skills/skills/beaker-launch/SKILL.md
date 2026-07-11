---
name: beaker-launch
description: Launch, size, and monitor training jobs on Beaker (AI Hub) for the disRNN stack — cluster allowlist, preferred cluster order, priority/preemption rules, GPU-bundle sizing, the resumable launcher, and validation. Use whenever submitting, debugging, or scheduling Beaker experiments or W&B sweeps on AI Hub clusters.
---

# Launching on Beaker (AI Hub)

Canonical detail: `docs/beaker-playbook.md` (scheduling rules) and
`code/beaker/README.md` (flow, cluster table, memory pitfalls, resumable mechanics).
**If this skill and those docs conflict, the docs win — read them for any non-trivial launch.**

## Hard rules first

1. **Submit ONLY to `hub` clusters** (`octo-hub-*`, `octo.hub-*`, `aihub-*`).
   **NEVER** to non-hub clusters (`aipbd-*`, `siti-*`, `dev-*`, other `octo.ai-*`)
   even if idle — they belong to other science units.
   Sole verified exception: `ai1/octo.ai-aws-g6e` accepts our **low-priority
   preemptible** jobs only (AWS, reaches S3, same L40S bundle as `octo-hub-aws-l40s`).
2. **Never run the launch's compute on the login node** — the launcher itself is fine
   (it only submits), the training is not.
3. Use the `disrnn-cpu` conda env for `wandb`/`beaker`/YAML tooling:
   `conda activate disrnn-cpu` (`/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-cpu`).
4. Workspace/budget: `WS=ai1/aind-dynamic-foraging-foundation-model`.

## Check available resources FIRST (mandatory for large jobs)

**Before launching any large job (> 4 GPUs / > 4 concurrent tasks), run the
capacity check and route to a backend that actually has schedulable GPUs.**
This is a hard rule (AGENTS.md §10) — do not assume the preferred cluster order
below has free slots.

```bash
# schedulable = free AND not on a cordoned node (Beaker) / Cfg-Alloc on non-drain nodes (HPC)
python code/check_gpu_availability.py            # both backends
python code/check_gpu_availability.py --beaker   # Beaker only (no VPN needed)
python code/check_gpu_availability.py --hpc      # HPC only (needs Allen network / VPN)
```

Why the built-in counts lie:

- **Beaker** advertises a node's `free.gpu_count` even when the node is
  **`cordoned`** — those GPUs are *not* schedulable. A cluster can read "16 free"
  while all 16 sit on cordoned nodes (0 launchable). `check_gpu_availability.py`
  subtracts cordoned-node GPUs; `beaker cluster list ai1` does **not**.
- Always compare against the **schedulable** column, never raw "free".

If all Beaker clusters show 0 schedulable, **check HPC** (`--hpc`) and route the
job there instead (`hpc-launch` skill) — the two backends load-balance. HPC needs
VPN/Allen-network; if VPN is down, HPC is unreachable and Beaker is the only option
(wait out the queue — preemptible jobs burst as capacity frees / nodes uncordon).

## Cluster choice

Clusters (pick by *live schedulable capacity* first, then by these properties):

- `ai1/octo.ai-aws-g6e` — L40S 48GB, many slots; **low/preemptible only** (the
  verified non-hub exception).
- `ai1/octo-hub-aws-l40s` — L40S 48GB, same class.
- `ai1/octo-hub-onprem-h200` — H200 141GB. **Use only when a task needs the memory**
  (wide `hidden_size=256` OOMs a 48GB L40S). **H200 is NOT inherently faster than
  L40S/g6e** for our workloads — do not prefer it on speed grounds; for narrow N it
  offers no throughput advantage and is often the most contended.

**GCP clusters cannot reach AWS S3** (`aind-scratch-data` DNS fails cross-cloud) —
never route DB/S3-backed jobs there.

## Priority & preemption (hard-won, verified 2026-06-22)

- Fan-outs: `{priority: low, preemptible: true}` — low bursts onto idle GPUs *beyond*
  the unallocated budget; `normal`+preemptible is capped at it (tasks pend while GPUs idle).
- `autoResume` is auto-applied to preemptible jobs — **never set it explicitly**
  (spec rejects `preemptible` + `autoResume`).
- Guaranteed slot (never evicted): `{priority: normal, preemptible: false}`.
- Tasks pending while physical slots are free ⇒ budget cap or GPU over-assignment, not capacity.

## GPU-bundle sizing (avoid silent multi-GPU grabs)

GPUs come bundled with host CPU/RAM (L40S ≈ 93 GiB + 12 CPU per GPU). Requesting more
`memory`/`cpuCount` than one bundle makes a `gpuCount: 1` job grab **multiple GPUs**
(e.g. `memory: 256GiB` → 3 GPUs). Size to one bundle: `--memory 90GiB --cpu 12` for
1 L40S GPU. Verify on the first scheduled job: `beaker job get` GPUS column /
`BEAKER_ASSIGNED_GPU_COUNT`.

## Preferred launch route: resumable pseudo-sweep

For grid sweeps of long preemptible runs, use `launch_beaker_resumable.py` — it expands
a `method: grid` sweep into one self-contained, checkpoint-resumable Beaker task per
grid point (no sweep controller; grid-only):

```bash
conda activate disrnn-cpu
WS=ai1/aind-dynamic-foraging-foundation-model
python code/launch_beaker_resumable.py \
  --sweep studies/<study>/variants/<variant>/sweep.yaml \
  --experiment studies/<study>/variants/<variant>/experiment.yaml \
  --workspace "$WS" \
  --label <short-label> \
  --note "why this run exists + what we want to learn"
# --no-submit renders the spec for inspection without launching
```

It sets the W&B group to `<variant>@<launch_id>` and injects `DISRNN_META_*`
provenance (see the study-conventions skill). Requires
`training.checkpoint_every_n_steps > 0` for resume to work.

Native alternative (real `wandb agent` sweep, not preemption-resilient):
`python code/launch_beaker.py --sweep <sweep.yaml> --experiment <experiment.yaml>`.

Per-task cluster/resource splits aren't supported by the launcher — render with
`--no-submit`, edit `constraints.cluster`/`resources` per task, then
`beaker experiment create -w "$WS" <spec>.yaml`.

## Validate, then fan out

Validate one unit first **only when something is untested** (new cluster, new sizing,
changed spec); check assigned GPUs/resources on the first scheduled job before trusting
the fan-out. Routine repeats of known-good launches: fan out directly.

## Monitoring & debugging

- `https://beaker.org/ex/<id>`; runs appear in the study's W&B project.
- `beaker experiment get <id> --format json`, `beaker job get <id>`,
  `beaker cluster get <cluster> --format json`.
- When explaining scheduling/quota behavior, **pull the JSON and cite the field**;
  label "verified:" vs "likely, unconfirmed:" (AGENTS.md §11).
- After the launch settles, write `launch_record_<label>/results.md`
  (see posthoc-reporting skill).
