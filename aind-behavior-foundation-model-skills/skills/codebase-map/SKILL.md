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
- **Claude Science layer** (AGENTS.md §13): the agent's persistent brain runs on the
  user's Mac; GitHub is the source of truth, tracked by the Mac authoring clone
  (`~/Scripts/aind-disrnn-dispatcher`) and a pull-only HPC runtime checkout
  (`/home/han.hou/code/...`). Load balancing: CPU jobs → HPC SLURM, GPU jobs →
  Beaker. Full scheme + credentials: `references/claude-science-workflow.md`.

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
| `code/check_gpu_availability.py` | Schedulable-GPU probe (Beaker + HPC) — mandatory before large launches (AGENTS.md §10) |
| `code/beaker_client.py` | Sandbox-safe Beaker/W&B client helpers used by the launchers |
| `studies/<study>/` | One folder per scientific question (`NN-{model}-{purpose}`); variants + analysis (see study-conventions skill) |
| `studies/util/` | Shared analysis helpers: `_meta.py` (provenance block), `plot_style.py` |
| `docs/` | Canonical deep-dive docs (see index below) |
| `AGENTS.md` | Behavioral guardrails — the terse rules; docs hold the detail |

## Where knowledge lives (one canonical home per topic)

- **AGENTS.md** — always-loaded terse guardrails (both repos).
- **This skills pack** — canonical for cross-cutting *operational* knowledge:
  launching (beaker-launch, hpc-launch), study conventions (study-conventions),
  reporting (posthoc-reporting), the Claude Science workflow
  (`references/claude-science-workflow.md`). The former `docs/*.md` playbooks were
  absorbed into these skills and are now pointer stubs.
- **Code-adjacent living docs** — canonical for code-coupled reference; skills
  defer to them:
  - `../aind-disrnn-wrapper/code/TRAINING.md` — **§1.5 "Run lifecycle & key
    switches" first**: the four run phases, the `_step` warmup offset, the **two
    different held-out switches**, checkpoints/resumability/extendability.
    **Read before interpreting any run's logs or metrics** (distilled in the
    wrapper-runtime skill).
  - `../aind-disrnn-wrapper/code/POST_TRAINING_ANALYSIS.md` — the analysis
    codebase + `run_analysis.py` CLI.
  - `../aind-disrnn-wrapper/beaker/README.md` — image build plane +
    GPU-efficiency benchmarks (why L40S beats H200 here; batch/length-bucketing
    levers).
  - `code/beaker/README.md` — Beaker flow, cluster + **image** tables, memory
    pitfalls, resumable-run mechanics.
  - `code/hpc/README.md` — SLURM setup, launch variants, monitoring.

## Non-negotiable rules (from AGENTS.md)

- **Never run heavy work on the login node** — always `srun`/`sbatch` (HPC) or Beaker.
- Beaker: **submit only to `hub` clusters** (+ the one verified `octo.ai-aws-g6e`
  low/preemptible exception) — see the beaker-launch skill.
- **Check schedulable capacity before any large launch** (> 4 GPUs / > 4 concurrent
  tasks): `python code/check_gpu_availability.py` — raw `beaker cluster list`/`sinfo`
  counts include cordoned/drained nodes and lie.
- Control-plane conda env on Allen HPC: `disrnn-cpu`
  (`/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-cpu`), not base.
- Conventional Commits; never squash-merge PRs (`gh pr merge <n> --merge`).
- Human-facing logs: Seattle time (`TZ=America/Los_Angeles`), include W&B links.
- Verify infra claims with data (`beaker ... --format json`, W&B API) before asserting;
  label "verified:" vs "likely, unconfirmed:".

## Which skill next

- Launch on Beaker / AI Hub → **beaker-launch**
- Launch on Allen on-prem SLURM → **hpc-launch**
- Interpret a run's logs/metrics, held-out numbers, checkpoints/resume, or the
  training/analysis code itself → **wrapper-runtime**
- Set up a new study/variant or name W&B groups → **study-conventions**
- Write or regenerate analysis reports → **posthoc-reporting**
