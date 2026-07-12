---
name: beaker-launch
description: Launch, size, and monitor training jobs on Beaker (AI Hub) for the disRNN stack — cluster allowlist, preferred cluster order, priority/preemption rules, GPU-bundle sizing, the resumable launcher, extend/restore (`restore_from_run_id`) and held-out re-scoring, and validation. Use whenever submitting, debugging, or scheduling Beaker experiments or W&B sweeps on AI Hub clusters.
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

## Resuming, extending & re-scoring runs

Three distinct mechanisms — don't conflate them (full lifecycle detail: wrapper
`../aind-disrnn-wrapper/code/TRAINING.md` §1.5 "Run lifecycle & key switches"):

1. **Preemption resume — automatic, WITHIN one experiment.** Covered under
   "Priority & preemption" above: a preempted `preemptible: true` task restarts as
   the *same* task with the *same* `/results` dataset, re-finds its latest
   `checkpoints/step_<N>/train_state.pkl`, and continues (skipping warmup). Needs
   `checkpoint_every_n_steps > 0` + the trainer gate `training.auto_resume` (default; distinct
   from Beaker's own `autoResume` spec field above). No flags, no new
   experiment.
2. **Extend a finished run to a longer horizon — ACROSS experiments.** Launch a
   *new* experiment with `model.training.restore_from_run_id=<source W&B run name>`
   (or per-cell env `DISRNN_RESTORE_FROM_RUN_ID` — env wins, so a sweep can pass a
   per-cell id) and a **larger** `n_steps`. Before training, the entrypoint downloads
   the source run's `<mtype>-output-<run_id>:latest` artifact (`mtype` ∈
   {`disrnn`,`gru`}) into `outputs/`, so the trainer resumes from its checkpoint and
   skips warmup. Trainer-agnostic. **Prereq: the source run must have FINISHED** — its
   `training-output` artifact is uploaded once at end of training (not per checkpoint),
   so in-progress runs cannot be extended. Fails **loudly** if the artifact is missing —
   never silently restarts from scratch.
3. **Re-score a finished run's held-out stage only — no re-training.**
   `python resume_heldout_beaker.py --run-id <wandb_run_id>` (wrapper repo root, inside
   a Beaker container that reaches GCS + W&B). Runs the held-out fine-tune ONLY off the
   downloaded checkpoint tree, reads every knob (seed, `checkpoint_policy`, held-out set,
   finetune `n_steps`/`lr`) from the SOURCE run's own config, and re-injects `heldout/*`
   back into the ORIGINAL W&B run. Use it to backfill metrics added *after* a run trained
   (e.g. the 3-way ignore-class precision/recall/F1/PR-AUC). This is the
   **exact-reproduction** path: unlike (2)'s restore — which resumes the training
   entrypoint and redraws a fresh held-out set off the restored checkpoints — this
   reproduces the source run's original held-out numbers. (Beaker port of the HPC
   `resume_heldout.py`.)

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

## Launching from the Claude Science Mac sandbox

The launchers (`code/launch_beaker.py`, `code/beaker_client.py`,
`code/launch_beaker_resumable.py`) run **directly from the Mac sandbox** — no HPC
hop. They are sandbox-safe: `create_wandb_sweep()` hits the W&B GraphQL API
directly (no `wandb-core` subprocess), and `get_beaker_client()` builds
`beaker.Beaker` from `Config(user_token=os.environ["BEAKER_TOKEN"])` directly (no
`~/.beaker/config.yml`). Creds `BEAKER_TOKEN` + `WANDB_API_KEY` are in the sandbox
env; `beaker.org` and `api.wandb.ai` each need a one-time `request_network_access`
grant.

```bash
cd code
# PYTHONSAFEPATH=1 in the sandbox drops the script dir from sys.path, so the
# launcher's sibling imports (beaker_client, ...) fail with ModuleNotFoundError
# unless code/ is on PYTHONPATH:
PYTHONPATH="$(pwd):$PYTHONPATH" python launch_beaker.py \
  --sweep beaker/sweep_mvp.yaml --experiment beaker/experiment_mvp.yaml \
  --workspace ai1/aind-dynamic-foraging-foundation-model \
  --output-dir ./out --label <label> --note "why this run exists" \
  --no-submit    # SAFE dry-run: creates the W&B sweep (prints SWEEP_ID) and
                 # renders the spec, does NOT submit. Drop it to actually submit.
```

`launch_beaker.py` is two-step: create the W&B sweep (prints `SWEEP_ID` =
`entity/project/id`), then submit the Beaker experiment unless `--no-submit`.
Full recipe: `docs/claude-science-workflow.md` -> "Mac -> Beaker launch".

**Verify the image name before submitting.** Old example specs
(`experiment_h100.yaml`, `experiment_h200.yaml`, `experiment_pack.yaml`) reference
`beaker: han-hou/disrnn-wrapper`, which **no longer exists** ->
`ImageNotFound`/404. The current image for the `ai_hub_pck_integration` line is
`han-hou/disrnn-wrapper-pck-integration`. List live images and set the spec's
`image.beaker` to one that exists:

```python
# repl cell, via beaker-py
[im.full_name for im in b.workspace.images(
    workspace="ai1/aind-dynamic-foraging-foundation-model")]
# or CLI: beaker workspace images ai1/aind-dynamic-foraging-foundation-model
```

Code is pulled fresh at container startup (`entrypoint.sh` checks out
`WRAPPER_REF`/`DISPATCHER_REF`), so **code/config edits need no image rebuild** —
push the branch and set the refs. Rebuild only when pinned dependencies change.

**Transient node failure != code bug.** A job can die in ~5 s with
`status.message: "no space left on device"` and `started=None` — the node's NVMe
filled while `mkdir`-ing the dataset dir (seen on `gcp-h100`). This is a per-node
infra failure, not your code. Confirm with `beaker job get <id> --format json`
(`status.message` / `status.started`), then just resubmit — it lands elsewhere.

**Pre-launch capacity check.** Before a large fan-out (>4 GPUs / >4 concurrent
tasks), run `python code/check_gpu_availability.py --beaker` (AGENTS.md §10) —
reports GPUs that are free **and** not cordoned, by type. That script lands with
the `feat/ignore-trials-scaling` merge; where it's absent, fall back to
`beaker cluster list ai1`.

