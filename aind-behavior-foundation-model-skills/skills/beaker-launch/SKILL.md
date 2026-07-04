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

## Cluster choice

Preferred order for known-good low/preemptible S3-backed jobs:

1. `ai1/octo.ai-aws-g6e` — L40S, many slots, faster than H200 for our workloads;
   **low/preemptible only** (the verified exception).
2. `ai1/octo-hub-onprem-h200` — many slots; needed for big-GPU jobs (hidden_size=256
   needs H200's 141 GB; it OOMs a 48 GB L40S).
3. `ai1/octo-hub-aws-l40s` — same L40S class, more contended.

**GCP clusters cannot reach AWS S3** (`aind-scratch-data` DNS fails cross-cloud) —
never route DB/S3-backed jobs there. Check free slots: `beaker cluster list ai1`.

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
