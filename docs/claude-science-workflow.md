---
aliases:
  - claude science workflow
  - mac orchestration
  - agent workflow
  - where jobs run
tags:
  - convention
  - meta
  - launching
  - infrastructure
---

# Claude Science workflow

How this repo is driven from **Claude Science** (the agent running on the user's
Mac, 24/7). The one-line rule lives in `AGENTS.md` §13; this file holds the full
scheme. Complements — does not replace — the HPC (`hpc-launch`) and Beaker
(`beaker-launch`) launch skills, which cover the mechanics of each launcher.

## Roles

- **Mac (Claude Science)** — the orchestration brain. Holds persistent project
  memory, artifacts, and planning; drives everything else. Runs continuously, so
  it is the durable state, not any remote shell session.
- **GitHub** (`AllenNeuralDynamics/aind-disrnn-dispatcher`) — the single source of
  truth for code. Both checkouts below track it.
- **HPC login node** (`hpc-code.corp.alleninstitute.org`) — a **pull-only runtime
  checkout** at `/home/han.hou/code/aind-disrnn-dispatcher`. Synced with
  `git pull`, never edited in place. Reached over SSH (VPN when remote).
- **Mac clone** (`~/Scripts/aind-disrnn-dispatcher`) — the **authoring checkout**,
  sibling to `aind-disrnn-wrapper`. Where the agent edits source and shows diffs.

## Where each task runs

| Task | Runs on | Needs VPN? |
| --- | --- | --- |
| GPU training → Beaker | Mac (beaker CLI/API + token) | No |
| CPU training → HPC SLURM | HPC login node over SSH (`git pull` to sync first) | Yes, only here |
| Rebuild the `disrnn-wrapper` image | Mac (Docker; HPC has none) | No |
| W&B post-hoc analysis | Mac sandbox (GraphQL) or HPC (SDK) | No |
| Edit / review repo code | Mac clone, native diffs | No |

Rationale for the CPU→HPC / GPU→Beaker split: first-order load balancing —
HPC has many CPU nodes, Beaker has the better GPUs. Both launchers
(`launch_hpc.py`, `launch_beaker*.py`) live in this one repo and both only
*submit* (training runs remotely), so a single checkout drives both; no code is
duplicated. All results upload to W&B with provenance, which is the basis for
post-hoc analysis (`docs/posthoc-analysis.md`).

## Division of labor: agent authors, user owns git

The Claude Science sandbox **cannot create a `.git` directory** inside a granted
host path (`mkdir <granted>/.git` → "Operation not permitted"). Consequences:

- **Cloning** a repo into `~/Scripts` must be done by the user in their own
  terminal (their local git is also the one that carries SAML-SSO authorization
  for the org).
- **Editing tracked files, committing, and pushing** into an *existing* checkout
  all work through the host grant — the block is only on creating `.git`, not on
  writing into one.

So the agent edits files and shows `git diff`; commit/push can be done by either
party. The user always owns the initial clone and the SSO auth.

## Reading W&B from the sandbox

The `wandb` Python SDK's `wandb.Api()` fails in the sandbox (it spawns a
`wandb-core` subprocess and writes under `~/.config/wandb`, both blocked). Pull
run/sweep data via the GraphQL endpoint with the `WANDB_API_KEY` credential
instead — see the `posthoc-reporting` skill for the snippet. `api.wandb.ai` and
`beaker.org` must each be allowlisted once (`request_network_access`).

## Mac → Beaker launch (copy-paste recipe)

The launchers (`code/launch_beaker.py`, `code/beaker_client.py`,
`code/launch_beaker_resumable.py`) are **sandbox-safe** — they run directly from
the Claude Science Mac sandbox, no HPC hop needed:

- `create_wandb_sweep()` hits the W&B GraphQL API directly (no `wandb-core`
  subprocess, no `~/.config/wandb`).
- `get_beaker_client()` builds `beaker.Beaker` from
  `Config(user_token=os.environ["BEAKER_TOKEN"])` directly — no `~/.beaker/config.yml`.

**Prereqs (one-time):**
- Credentials already in the sandbox env: `BEAKER_TOKEN`, `WANDB_API_KEY`.
- Network: `beaker.org` and `api.wandb.ai` each need a one-time
  `request_network_access` grant.

**PYTHONPATH quirk (sandbox only):** `PYTHONSAFEPATH=1` disables Python's normal
"script's own dir goes on `sys.path`" behavior, so the launchers' sibling imports
(`beaker_client`, etc.) fail with `ModuleNotFoundError`. Run them with the repo
`code/` dir on `PYTHONPATH`:

```bash
cd code
PYTHONPATH="$(pwd):$PYTHONPATH" python launch_beaker.py \
  --sweep    beaker/sweep_mvp.yaml \
  --experiment beaker/experiment_mvp.yaml \
  --workspace ai1/aind-dynamic-foraging-foundation-model \
  --output-dir ./out --label <short-label> --note "why this run exists" \
  --no-submit                # SAFE dry-run: creates the W&B sweep + renders the
                             # Beaker spec, does NOT submit. Drop it to submit.
```

`launch_beaker.py` is two-step: it (1) creates the W&B sweep and prints
`SWEEP_ID` (`entity/project/id`), then (2) unless `--no-submit`, submits the
Beaker experiment. `--no-submit` is the recommended first pass. (This quirk is
sandbox-only — on HPC/Code Ocean the launchers run without `PYTHONPATH`.)

**Image name — the #1 stale-fact trap.** Old example specs referenced
`beaker: han-hou/disrnn-wrapper`, which **no longer exists** →
`ImageNotFound` / 404. The current image for the `ai_hub_pck_integration` line is
`han-hou/disrnn-wrapper-pck-integration`. **List the current images before
launching** and set the experiment spec's `image.beaker` to a live one:

```python
# in a repl cell, via beaker-py
[im.full_name for im in b.workspace.images(
    workspace="ai1/aind-dynamic-foraging-foundation-model")]
# or CLI: beaker workspace images ai1/aind-dynamic-foraging-foundation-model
```

**Code edits need no image rebuild.** `beaker/entrypoint.sh` (in the wrapper)
does `git fetch origin $WRAPPER_REF && checkout FETCH_HEAD` (same for
`$DISPATCHER_REF`) at container startup. So pure code/config changes take effect
by pushing the branch and setting `WRAPPER_REF` / `DISPATCHER_REF` in the
experiment spec — **rebuild the image only when pinned dependencies change.**

**Pre-launch check (fact 5).** Before a large fan-out (>4 GPUs / >4 concurrent
tasks), run the schedulable-GPU probe `python code/check_gpu_availability.py
--beaker` (per AGENTS.md §10) — it reports GPUs that are free **and** not on a
cordoned node, by type, so you route to a pool with real capacity. That script
lands here with the `feat/ignore-trials-scaling` merge; on branches where it is
not yet present, fall back to `beaker cluster list ai1` and pick a hub cluster
with free slots (see the `beaker-launch` skill / `code/beaker/README.md` for the
cluster table).

**Transient node failure (not a code bug).** A GPU job (e.g. on `gcp-h100`) can
fail in ~5 s with `status.message: "no space left on device"` and
`started=None` — the node's NVMe filled while `mkdir`-ing the dataset dir. This
is a per-node infra failure, **not** your code. Just resubmit; it lands on
another node. Don't debug the training code for this signature.

## Credentials (stored in Claude Science, never in the repo)

- `WANDB_API_KEY` — W&B GraphQL/API access. Entity `AIND-disRNN`, login `houhan`.
- `BEAKER_TOKEN` — Beaker CLI/API (`user_token` from `~/.beaker/config.yml`).
- `GITHUB_TOKEN` — SSO-authorized PAT for the org; used in-memory for push, never
  persisted into git config.
