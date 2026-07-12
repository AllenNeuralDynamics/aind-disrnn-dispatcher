# aind-behavior-foundation-model-skills

Agent Skills pack for the AIND behavior foundation model stack. Each skill under
`skills/<name>/` follows the open [Agent Skills](https://agentskills.io/specification)
format, so the pack is consumable by any spec-compliant tool (Claude Code, Claude
Science, the Claude API/Agent SDK, ...).

## Skills

| Skill | Use for |
|---|---|
| `codebase-map` | Orienting in the dispatcher/wrapper two-repo architecture + the Claude Science workflow |
| `beaker-launch` | Launching jobs on Beaker (AI Hub): cluster allowlist, capacity checks, sizing, resumable runs |
| `hpc-launch` | Launching on Allen on-prem SLURM via `launch_hpc.py` / sbatch |
| `wrapper-runtime` | The training/analysis runtime: run lifecycle, held-out switches, checkpoints, `run_analysis.py` |
| `study-conventions` | Study/variant folder layout, W&B group naming, provenance, study wrap-up |
| `posthoc-reporting` | Report/JSON contracts, launch records, regeneration rules |

## Structure & authoring convention (progressive disclosure)

Each skill is a folder:

```
skills/<name>/
  SKILL.md          # lean, stable core: hard rules, decision guides, common commands
  references/*.md   # deep dives loaded on demand (procedures, recipes, verified lessons)
```

Authoring rule: **new hard rules and commands go in `SKILL.md`; new lessons and
procedures go in a `references/` file** with a one-line pointer from `SKILL.md`.
This keeps the always-loaded core small while nothing is lost.

## Single source of truth

- **This pack is canonical for cross-cutting operational knowledge** (launching,
  scheduling, study conventions, reporting, the Claude Science workflow). The
  former `docs/*.md` playbooks in the dispatcher were absorbed here and are now
  pointer stubs — update the skill, not the stub.
- **Code-adjacent living docs stay canonical for code-coupled reference** and the
  skills defer to them: the wrapper's `code/TRAINING.md` +
  `code/POST_TRAINING_ANALYSIS.md` (update-with-the-code contract), and
  `code/beaker/README.md` / `code/hpc/README.md` in the dispatcher.
- `AGENTS.md` (both repos) stays the always-loaded terse guardrail layer; it
  points into this pack for detail.

## Import

**Claude Code (whole pack via the repo's plugin marketplace):**

```
/plugin marketplace add AllenNeuralDynamics/aind-disrnn-dispatcher
/plugin install aind-behavior-foundation-model-skills@aind-behavior-foundation-model
```

Inside this repo (and the wrapper repo) the pack is auto-enabled via the
checked-in `.claude/settings.json`.

**Other Agent Skills tools (e.g. Claude Science):** import the `skills/` directory —
each subfolder is a standard SKILL.md skill (with its `references/`).
