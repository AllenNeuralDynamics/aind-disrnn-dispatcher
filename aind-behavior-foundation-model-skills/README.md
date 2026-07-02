# aind-behavior-foundation-model-skills

Agent Skills pack for the AIND behavior foundation model stack. Each skill under
`skills/<name>/SKILL.md` follows the open [Agent Skills](https://agentskills.io/specification)
format, so the pack is consumable by any spec-compliant tool (Claude Code, Claude
Science, the Claude API/Agent SDK, ...).

## Skills

| Skill | Use for |
|---|---|
| `codebase-map` | Orienting in the dispatcher/wrapper two-repo architecture |
| `beaker-launch` | Launching jobs on Beaker (AI Hub): cluster allowlist, sizing, resumable runs |
| `hpc-launch` | Launching on Allen on-prem SLURM via `launch_hpc.py` / sbatch |
| `study-conventions` | Study/variant folder layout, W&B group naming, provenance |
| `posthoc-reporting` | Report/JSON contracts, launch records, regeneration rules |

The skills are distilled operational guides; the canonical detail lives in this repo's
`docs/` and the `code/*/README.md` files — on conflict, those win.

## Import

**Claude Code (whole pack via the repo's plugin marketplace):**

```
/plugin marketplace add AllenNeuralDynamics/aind-disrnn-dispatcher
/plugin install aind-behavior-foundation-model-skills@aind-behavior-foundation-model
```

Inside this repo the pack is auto-enabled via the checked-in `.claude/settings.json`.

**Other Agent Skills tools (e.g. Claude Science):** import the `skills/` directory —
each subfolder is a standard SKILL.md skill.
