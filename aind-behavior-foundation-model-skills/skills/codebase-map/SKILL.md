---
name: codebase-map
description: Orient in the aind-disrnn-dispatcher codebase — the two-repo architecture (dispatcher = control plane, wrapper = training payload), Hydra config layout, where launchers/configs/docs/studies live, and which skill or doc to read next. Use when starting work in this repo, answering "where does X live", or deciding how to run/launch anything.
---

# Codebase map — aind-disrnn-dispatcher

## Big picture (two-repo architecture)

- **This repo (dispatcher)** is the *control plane* for the AIND-disRNN MLOps stack.
  It composes Hydra configs into job specs and submits them to one of three backends:
  Code Ocean, Beaker (AI Hub), or Allen on-prem SLURM HPC.
- **`aind-disrnn-wrapper`** (expected as a sibling checkout at
  `../aind-disrnn-wrapper`) is the *compute/runtime payload*: training code, the
  Beaker image, `run_hpc`. Job containers pull code fresh at startup, so code edits
  need **no image rebuild** (pin via `WRAPPER_REF` / `DISPATCHER_REF`).
- W&B (`entity: AIND-disRNN`) is the experiment tracker across all backends.

## Repo layout

| Path | What it is |
|---|---|
| `code/run_capsule.py` | Hydra entrypoint: composes `code/config/` into JSON job specs (see root `README.md` for override examples) |
| `code/config/` | Hydra config groups (`data=mice\|synthetic`, `model=disrnn\|baseline_rl\|...`) |
| `code/launch_beaker_resumable.py` | Preferred Beaker launcher: grid sweep → one resumable preemptible task per grid point |
| `code/launch_beaker.py` | Native-route Beaker launcher: `wandb sweep` + `wandb agent` replicas |
| `code/beaker/` | Beaker sweep/experiment YAMLs + `README.md` (control-plane detail) |
| `code/launch_hpc.py` | Allen on-prem SLURM launcher (W&B sweep + sbatch array) |
| `code/hpc/` | SLURM scripts, sweep YAMLs, `user.env`, + `README.md` |
| `code/launch_CO_wrapper.py` | Code Ocean route |
| `studies/<study>/` | One folder per scientific question; variants + analysis (see study-conventions skill) |
| `docs/` | Canonical deep-dive docs (see index below) |
| `AGENTS.md` | Behavioral guardrails — the terse rules; docs hold the detail |

## Doc index (canonical — these win over any skill summary)

- `docs/beaker-playbook.md` — cluster allowlist, scheduling, GPU-bundle sizing. **Read before any non-trivial Beaker launch.**
- `docs/study-organization.md` — studies/variants layout, W&B group naming, `meta` provenance.
- `docs/posthoc-analysis.md` — report/JSON contracts, Makefile, regeneration rules.
- `code/beaker/README.md` — Beaker flow, cluster table, memory pitfalls, resumable-run mechanics.
- `code/hpc/README.md` — SLURM setup, launch variants, monitoring.

## Non-negotiable rules (from AGENTS.md)

- **Never run heavy work on the login node** — always `srun`/`sbatch` (HPC) or Beaker.
- Beaker: **submit only to `hub` clusters** (+ the one verified `octo.ai-aws-g6e`
  low/preemptible exception) — see the beaker-launch skill.
- Control-plane conda env on Allen HPC: `disrnn-cpu`
  (`/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-cpu`), not base.
- Conventional Commits; never squash-merge PRs (`gh pr merge <n> --merge`).
- Human-facing logs: Seattle time (`TZ=America/Los_Angeles`), include W&B links.
- Verify infra claims with data (`beaker ... --format json`, W&B API) before asserting;
  label "verified:" vs "likely, unconfirmed:".

## Which skill next

- Launch on Beaker / AI Hub → **beaker-launch**
- Launch on Allen on-prem SLURM → **hpc-launch**
- Set up a new study/variant or name W&B groups → **study-conventions**
- Write or regenerate analysis reports → **posthoc-reporting**
