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

## Credentials (stored in Claude Science, never in the repo)

- `WANDB_API_KEY` — W&B GraphQL/API access. Entity `AIND-disRNN`, login `houhan`.
- `BEAKER_TOKEN` — Beaker CLI/API (`user_token` from `~/.beaker/config.yml`).
- `GITHUB_TOKEN` — SSO-authorized PAT for the org; used in-memory for push, never
  persisted into git config.
