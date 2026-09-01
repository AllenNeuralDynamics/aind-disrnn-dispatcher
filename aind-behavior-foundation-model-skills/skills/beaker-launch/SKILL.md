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
   **REVOKED 2026-08-22 (confirmed by Han):** `ai1/octo.ai-aws-g6e` and
   `ai1/octo.ai-aws-p5en` are **no longer available to us** and were formerly
   documented here as verified exceptions. Do not target them even as fallback
   entries — jobs silently never schedule. Observed the same day: g6e showed 0
   schedulable with **2075** queued jobs, p5en 1 schedulable with **1008** queued.
   Treat `references/scheduling-lessons.md`'s g6e/p5en guidance as historical.
   There are now **no** non-hub exceptions; submit only to `octo-hub-*` /
   `octo.hub-*` / `aihub-*`.
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
6. **Scientific submissions use immutable refs.** Templates may retain readable
   branch/tag values and inline comments, but the launchers require all three REF
   variables and resolve them to full SHAs before W&B sweep creation or Beaker
   submission. The rendered launch-record YAML is pinned. Mutable refs are only
   for direct smoke/development jobs; pin manually when bypassing the launchers.

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

- `ai1/octo-hub-aws-h200` — H200 141GB. Usable for S3-backed jobs.
- `ai1/octo-hub-onprem-h200` — H200 141GB, on-prem but **reaches AWS S3 fine**
  (verified 2026-08-22: a probe read the snapshot session table and printed
  `S3_PARQUET_OK sessions= 23868`). Often the emptiest pool, and study-01's H128
  column trained here. **H200 is NOT inherently faster than L40S** for our
  workloads — do not prefer it on speed grounds, prefer it on free slots.
- `ai1/octo-hub-aws-l40s` — L40S 48GB. Note wide `hidden_size=256` OOMs 48GB.

### The S3 rule — decides cluster choice before capacity does

**GCP clusters cannot reach the AWS S3 parquet cache**
(`s3://aind-scratch-data/aind-dynamic-foraging-cache`, us-west-2): intermittent
`Could not resolve hostname` / `SSL CA cert` `IOException`s mid-fetch. This rules
out `ai1/octo-hub-gcp-h100` and `ai1/octo.hub-gcp-h200` for **every mice-data
run**, however many GPUs they show free — and they frequently show the most free,
which is the trap. They are usable only for compute that touches no DB (e.g.
in-process synthetic data). Canonical list in `code/beaker/README.md`'s cluster
table; read it *before* running a capacity survey, not after.

**Usable for S3/DB-backed training:** `octo-hub-aws-h200`,
`octo-hub-onprem-h200`, `octo-hub-aws-l40s`.

### Don't let one study serialize itself

Live free-GPU count is not the same as *available to you*. A multi-variant study
can saturate a cluster with its **own** tasks, so later tasks queue behind
earlier ones and a parallel grid silently becomes serial (observed 2026-08-22:
11 running + 10 queued on `aws-h200`, all from one study, while
`onprem-h200` sat idle with 16 schedulable). Before launching a second variant,
check how many slots *your own* running tasks already hold, and spread variants
across the usable clusters.

If you split a grid across clusters, say so in the variant notes: the swept axis
and the cluster become **correlated**, so a cluster-level artifact would alias
onto that axis. Cheap mitigation — run one grid point on *both* clusters as a
cross-cluster check.

### Image ≠ pullable

`b.image.get(name)` succeeding only proves the image is **registered**, not that a
cluster can pull it. A stale image copied from an old study variant's
`experiment.yaml` failed to start on one cluster while working on another
(`Failed to start job: ... No such image: gcr.io/...: image not found`, ~2 min,
**no logs**, `exit_code=None` — the reason appears only on `job.status.message`).
When reusing an old template, re-point `image.beaker` at the current image and, on
an untried cluster, probe with a one-task job that echoes a marker before spending
a grid. Note `b.image.list()` does not exist in `beaker-py<2`; call
`b.image.get(name)` per candidate.

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

> **⚠️ Grids above ~15 tasks: Beaker rejects the single-experiment payload with a misleading 409.**
> Render with `--no-submit` first, check the resolved-JSON size, and split into ≤~15-task chunks
> submitted directly if it's too big — see `references/scheduling-lessons.md` "Resolved-JSON payload
> ceiling."

> **⚠️ Set `wandb.project=` in the sweep `command:` for resumable launches.** Unlike
> the native launcher (whose `wandb agent` sweep controller owns the project), the
> resumable launcher sets only the W&B **group**, not the project — each task runs
> `run_hpc` directly, so the project comes from Hydra's `wandb.project`, which
> **defaults to `test`**. Omitting it silently lands every run in the `test` project.
> Add `wandb.project=<study_project>` (and `wandb.entity=AIND-disRNN`) to the sweep's
> `command:` list, next to `wandb.tags`. The sweep's top-level `project:` field is read
> only by the native sweep controller, NOT by the resumable launcher.

> **⚠️ Pin `WRAPPER_REF`/`DISPATCHER_REF` to a full commit SHA, not a branch, for
> resumable/preemptible launches.** A preempted task auto-resumes by re-running
> `entrypoint.sh`, which re-checks-out the ref. A branch ref is mutable, so a resume can:
> (1) **fail** if the branch was deleted meanwhile — e.g. GitHub auto-deletes it on PR
> merge, so merging the fix branch mid-run breaks every resume; or (2) **silently run
> different code** than the first attempt if the branch advanced — the second half of a
> preempted run executes the new tip. A SHA is immutable: the resume re-checks-out the
> *same* commit, always resolvable. The launch record logs the commit for provenance, but
> that records *intent* — only pinning the ref to a SHA makes the *executed* code match it.
> The entrypoint already accepts a SHA (`git fetch --depth 1 origin <sha>`). Grab it at
> launch time from the branch you're about to run:
>
> ```bash
> git ls-remote origin <branch> | cut -f1   # -> the SHA to paste as WRAPPER_REF/DISPATCHER_REF
> ```
>
> (Resolve against the **remote** — the SHA must be pushed, since the container fetches
> from origin.) A branch ref is fine only for one-shot, non-resumable jobs (no resume
> window). If branch-pinning ever recurs as a footgun, teach the launcher to resolve the
> branch→SHA at submit time so you keep typing branch names but the task spec pins the SHA.

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
- **`exit_code == 0` is NOT proof the metrics landed.** On a heavily-preempted run W&B
  can mark the run `crashed` and silently drop its final summary write, leaving the
  task exit-0 on Beaker with **no `heldout/*` key at all** — invisible from either side
  alone, and never retried. Before calling a grid complete, reconcile the
  Beaker-finished count against the W&B-finished count and chase any persistent gap
  (`references/scheduling-lessons.md`); recovery is usually free from the surviving
  per-subject table (`references/resume-extend-rescore.md` mechanism 4).
- After the launch settles, write `launch_record_<label>/results.md`
  (see posthoc-reporting skill).
- **Any post-launch intervention — resubmit, rescue, probe, tier change — must write a
  launch record**, or it is lost: the launchers only record the *first* launch. Use
  `studies/util/launch_record.py::write_intervention(..., platform="beaker",
  job_refs=[{"type": "beaker_experiment", "id": exp.id}])`; schema and the wrap-up
  reconciler (`validate_provenance.py`) live in the `study-conventions` skill.

## References (read on demand)

- `references/sandbox-launch.md` — launching from the Claude Science Mac sandbox:
  PYTHONPATH quirk, **image-name verification** (the #1 stale-fact trap),
  transient node failures (resubmit, don't debug).
- `references/resume-extend-rescore.md` — the four distinct mechanisms:
  automatic preemption resume, extend a finished run (`restore_from_run_id`),
  re-score held-out only (`resume_heldout_beaker.py`), and **backfill a lost metric
  from its surviving table artifact** (no GPU — try before re-scoring; includes the
  verified trial-weighted-geometric aggregation).
- `references/scheduling-lessons.md` — the verified g6e exception, priority-tier
  measurements, bundle over-assignment, cross-cloud S3, the resolved-JSON payload
  ceiling, **exit-0-with-a-missing-metric**, verify-with-data.
