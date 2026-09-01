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
  Beaker image, `run_hpc`. Job containers refresh the wrapper, dispatcher, and
  `aind-dynamic-foraging-models` sources at startup, so code edits need **no image
  rebuild** (pin via `WRAPPER_REF`, `DISPATCHER_REF`, and `FORAGING_MODELS_REF`).
- W&B (`entity: AIND-disRNN`) is the experiment tracker across all backends.
- **Claude Science layer** (AGENTS.md §13): the agent's persistent brain runs on the
  user's Mac; GitHub is the source of truth, tracked by the Mac authoring clone
  (`~/Scripts/aind-disrnn-dispatcher`) and a pull-only HPC runtime checkout
  (`/home/han.hou/code/...`). Load balancing: CPU jobs → HPC SLURM, GPU jobs →
  Beaker. Full scheme + credentials: `references/claude-science-workflow.md`.

## Repo layout

| Path | What it is |
|---|---|
| `code/config/` | Hydra config groups (`data=mice\|synthetic`, `model=disrnn\|baseline_rl\|...`) — composed here, consumed by the wrapper's `run_capsule.py` |
| `code/launch_beaker_resumable.py` | Preferred Beaker launcher: grid sweep → one resumable preemptible task per grid point |
| `code/launch_beaker.py` | Native-route Beaker launcher: `wandb sweep` + `wandb agent` replicas |
| `code/beaker/` | Beaker sweep/experiment YAMLs + `README.md` (control-plane detail) |
| `code/launch_hpc.py` | Allen on-prem SLURM launcher (W&B sweep + sbatch array) |
| `code/hpc/` | SLURM scripts, sweep YAMLs, `user.env`, + `README.md` |
| `code/launch_CO_wrapper.py` | Code Ocean route |
| `code/check_gpu_availability.py` | Schedulable-GPU probe (Beaker + HPC) — mandatory before large launches (AGENTS.md §10) |
| `code/beaker_client.py` | Sandbox-safe Beaker/W&B client helpers used by the launchers |
| `studies/<study>/` | One folder per scientific question (`NN-{model}-{purpose}`); variants + analysis (see study-conventions skill) |
| `studies/util/` | Shared helpers: `_meta.py` (provenance block), `plot_style.py`, `launch_record.py` (intervention records), `validate_provenance.py` (wrap-up reconciler), `watch_runs.py` (poll Beaker+W&B, append timestamped snapshots so transitions between check-ins aren't lost) |
| `docs/` | Design records + pointer stubs (index below) |
| `AGENTS.md` | Behavioral guardrails — the terse rules; this pack and the code-adjacent docs hold the detail |

## Where knowledge lives (one canonical home per topic)

- **AGENTS.md** — always-loaded terse guardrails (both repos).
- **This skills pack** — canonical for cross-cutting *operational* knowledge:
  launching (beaker-launch, hpc-launch), study conventions (study-conventions),
  reporting (posthoc-reporting), git/provenance (git-session-isolation), the Claude
  Science workflow (`references/claude-science-workflow.md`). The former `docs/*.md`
  playbooks were absorbed into these skills and are now pointer stubs — update the
  skill, not the stub.
- **`docs/` design records** — canonical for *forward-looking modeling decisions*, which
  no skill covers. Read the relevant one before touching that modeling direction:
  - `docs/design-hb-baseline.md` — settled decision record for the hierarchical-Bayesian
    cognitive-model baseline (implementation state lives in dispatcher issue #72, not the
    note). Relevant to `studies/08-hb-vs-gru-heldout/`.
  - `docs/design-hierarchical-vi-foundation-model.md` — design/plan (not yet implemented)
    for a hierarchical mixed-effects foundation model via amortized VI.
  - `docs/repo-split-plan.md` — why the dispatcher/wrapper split is shaped as it is; read
    before moving code across the two repos.
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

## Which skill next — and why the rules are not on this page

**This map answers "where is X" and "which skill next"; it deliberately does not state
the rules themselves.** Each rule has one owner, and a copy here is a second copy that
rots while the owner is corrected — this section used to carry the Beaker cluster
allowlist and still listed two revoked non-hub clusters as verified exceptions weeks
after `beaker-launch` *and* `code/beaker/README.md` were both fixed. Since this is the
first skill loaded, that stale copy was the first thing every agent read.

| To do / know | Load |
|---|---|
| Submit to Beaker: cluster allowlist, capacity, sizing, priority, resumable launches | **beaker-launch** |
| Launch on Allen on-prem SLURM; the login-node prohibition; `disrnn-cpu` | **hpc-launch** |
| Interpret logs/metrics, held-out numbers, checkpoints/resume; training + analysis code | **wrapper-runtime** |
| Create a study/variant, name W&B groups, record an intervention | **study-conventions** |
| Write or regenerate reports, figures, curated JSON | **posthoc-reporting** |
| Commit / branch / push, provenance SHAs, concurrent sessions on a shared checkout | **git-session-isolation** |
| File an issue, set its project-board fields, link a PR | **issue-tracking** |
| Commit style, merge policy, Seattle-time logs, verify-with-data | `AGENTS.md` (always loaded) |

Two `AGENTS.md` rules are worth repeating anywhere, because breaking them costs a job or
a dataset rather than a correction: **never run heavy work on the login node**, and
**check schedulable capacity before any large launch** — raw `beaker cluster list` /
`sinfo` counts include cordoned and drained nodes.
