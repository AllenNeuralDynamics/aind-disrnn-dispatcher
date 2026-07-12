---
name: beaker-launch
description: Launch, size, and monitor training jobs on Beaker (AI Hub) for the disRNN stack — cluster allowlist, capacity checking, priority/preemption rules, GPU-bundle sizing, the resumable launcher, extend/restore and held-out re-scoring, and validation. Use whenever submitting, debugging, or scheduling Beaker experiments or W&B sweeps on AI Hub clusters.
---

# Launching on Beaker (AI Hub)

This skill is the source of truth for Beaker launching (hard rules mirrored from
`AGENTS.md` §10, which wins on conflict). Deep detail: `references/` here and
`code/beaker/README.md` (flow, cluster + **image** tables, memory pitfalls,
resumable mechanics).

## Hard rules first

1. **Submit ONLY to `hub` clusters** (`octo-hub-*`, `octo.hub-*`, `aihub-*`).
   **NEVER** to non-hub clusters (`aipbd-*`, `siti-*`, `dev-*`, other `octo.ai-*`)
   even if idle — they belong to other science units.
   Verified exceptions: `ai1/octo.ai-aws-g6e` (L40S) and `ai1/octo.ai-aws-p5en`
   (H200 141 GB) accept our **low-priority preemptible** jobs only
   (see `references/scheduling-lessons.md`).
2. **Never run the launch's compute on the login node** — the launcher itself is fine
   (it only submits), the training is not.
3. Use the `disrnn-cpu` conda env for `wandb`/`beaker`/YAML tooling:
   `conda activate disrnn-cpu` (`/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-cpu`).
   **It needs `beaker-py<2`** — the launchers and `check_gpu_availability.py` do
   `from beaker import Beaker, Config`, and beaker-py 2.x dropped the `Config`
   export (`ImportError: cannot import name 'Config'`). If beaker-py is missing
   entirely (it was on HPC until 2026-07-11), `pip install "beaker-py<2"`.
   The `beaker` CLI is not a substitute: the launchers use beaker-py directly so
   the same code path works in the Mac sandbox, which has no CLI.
4. Workspace/budget: `WS=ai1/aind-dynamic-foraging-foundation-model`.
5. From HPC/sandbox, **pass `--output-dir`** to the launchers (or rely on the
   repo-local `results/` fallback) — `/results` is the Code Ocean path.
   Credentials on HPC: `BEAKER_TOKEN` is *not* in the env; read it from
   `~/.beaker/config.yml` (`user_token`). `WANDB_API_KEY` likewise comes from
   `~/.netrc`.

## Check available resources FIRST (mandatory for large jobs)

**Before launching any large job (> 4 GPUs / > 4 concurrent tasks), run the
capacity check and route to a backend that actually has schedulable GPUs**
(AGENTS.md §10) — do not assume any cluster has free slots.

```bash
# schedulable = free AND not on a cordoned node (Beaker) / Cfg-Alloc on non-drain nodes (HPC)
python code/check_gpu_availability.py            # both backends
python code/check_gpu_availability.py --beaker   # Beaker only (no VPN needed)
python code/check_gpu_availability.py --hpc      # HPC only (needs Allen network / VPN)
```

Raw counts lie: Beaker advertises `free.gpu_count` on **cordoned** nodes (not
schedulable), and `sinfo` counts `drain`/`down` nodes — the script strips both.
If all Beaker clusters show 0 schedulable, route to HPC (`hpc-launch` skill); the
two backends load-balance. If VPN is down, HPC is unreachable and Beaker is the
only option (preemptible jobs burst as capacity frees / nodes uncordon).

## Cluster choice

Pick by **live schedulable capacity first**, then by these properties:

- `ai1/octo.ai-aws-g6e` — L40S 48GB, many slots; **low/preemptible only** (a
  verified non-hub exception).
- `ai1/octo-hub-aws-l40s` — L40S 48GB, same class.
- `ai1/octo-hub-onprem-h200` — H200 141GB. **Use only when a task needs the memory**
  (wide `hidden_size=256` OOMs a 48GB L40S). **H200 is NOT inherently faster than
  L40S/g6e** for our workloads — do not prefer it on speed grounds.
- `ai1/octo.ai-aws-p5en` — H200 141GB (8/node); **low/preemptible only** (the other
  verified non-hub exception). The preemptible route to H200 memory when the on-prem
  H200 pool is full.

**GCP clusters cannot reach AWS S3** — never route DB/S3-backed jobs there.

## Priority & preemption

- Fan-outs: `{priority: low, preemptible: true}` — low bursts onto idle GPUs *beyond*
  the unallocated budget; `normal`+preemptible is capped at it (tasks pend while GPUs idle).
- `autoResume` is auto-applied to preemptible jobs — **never set it explicitly**
  (spec rejects `preemptible` + `autoResume`).
- Guaranteed slot (never evicted): `{priority: normal, preemptible: false}`.
- Mechanism detail + verified measurements: `references/scheduling-lessons.md`.

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

## References (read on demand)

- `references/sandbox-launch.md` — launching from the Claude Science Mac sandbox:
  PYTHONPATH quirk, **image-name verification** (the #1 stale-fact trap),
  transient node failures (resubmit, don't debug).
- `references/resume-extend-rescore.md` — the three distinct mechanisms:
  automatic preemption resume, extend a finished run (`restore_from_run_id`),
  re-score held-out only (`resume_heldout_beaker.py`).
- `references/scheduling-lessons.md` — the verified g6e exception, priority-tier
  measurements, bundle over-assignment, cross-cloud S3, verify-with-data.
