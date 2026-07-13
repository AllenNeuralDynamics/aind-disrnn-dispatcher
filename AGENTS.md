# Global AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

The four-principle backbone (Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution) is adapted from [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills), which distills [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls. HPC-specific rules and the commit-message convention below are local additions.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler approach exists, say so.
- If something is unclear, stop and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No configurability that was not requested.
- No error handling for impossible scenarios.
- If the solution is overcomplicated, simplify it.

## 3. Surgical Changes

Touch only what is needed. Clean up only your own mess.

When editing existing code:
- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor unrelated code.
- Match existing style.
- If you find unrelated dead code, mention it but do not delete it.

When your changes create orphans:
- Remove imports, variables, or functions made unused by your change.
- Do not remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the request.

## 4. Goal-Driven Execution

Define success criteria and verify.

Transform tasks into verifiable goals:
- Add validation -> write failing tests for invalid inputs, then make them pass.
- Fix a bug -> write a reproducing test, then make it pass.
- Refactor -> ensure tests pass before and after.

For multi-step tasks, use a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These guidelines are working when diffs contain fewer unnecessary changes, solutions are simpler, and clarifications happen before implementation.

## 5. HPC Execution Safety

Never run computation-intensive work on the login node (where the agent runs).

- Always use `srun` or `sbatch` for heavy workloads.
- This includes training jobs, sweeps, and tests.

## 6. Semantic Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format for every commit:

```text
<type>(<optional scope>): <short imperative summary>

<optional body explaining what and why, wrapped at ~72 chars>

<optional footer, e.g. "Refs #123" or "BREAKING CHANGE: ...">
```

Allowed `<type>` values:

- `feat` — new user-visible feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `perf` — performance improvement
- `test` — add or fix tests
- `build` — build system, dependencies, or packaging
- `ci` — CI configuration or scripts
- `chore` — maintenance, tooling, or housekeeping with no src/test impact
- `revert` — revert a prior commit

Rules:

- Summary line in the imperative mood, no trailing period, <= 72 chars.
- One logical change per commit; split unrelated changes into separate commits.
- Use `<scope>` for the affected area when helpful (e.g. `feat(launcher): ...`, `docs(readme): ...`).
- Mark breaking changes with `!` after the type/scope (e.g. `feat(api)!: ...`) and a `BREAKING CHANGE:` footer.
- Body explains the motivation and any non-obvious consequences; don't restate the diff.

## 7. Human-Facing Logs & Reporting

When logging progress or writing anything for a human to read (status updates, README
status logs, run reports, PR/commit notes), make it directly readable for the user:

- **Use Seattle time** (`America/Los_Angeles`), not UTC. Stamp times like `10:48 PT`
  (`TZ=America/Los_Angeles date`). Avoid forcing the reader to convert from UTC.
- **Link the W&B sweep/run.** When reporting a metric or run, include its W&B link so the
  reader can click through (e.g. the project `https://wandb.ai/<entity>/<project>` or the
  specific run URL). Beaker experiments aren't W&B sweeps — link the W&B project/run, and
  the Beaker experiment id/link when relevant.

## 8. Study & Experiment Organization

One folder per study under `studies/<name>/`; variants as self-contained subfolders
`variants/<variant>/`; one W&B project per study, one group per variant
(`<variant>@<launch_id>`). A study answers one scientific question — variants are not
separate studies. Full conventions + the provenance/`meta` scheme: the
**`study-conventions` skill** (`aind-behavior-foundation-model-skills/skills/study-conventions/`).

## 9. Merging Pull Requests

**Never squash-merge a PR.** Merge with a **merge commit** (no fast-forward, e.g.
`gh pr merge <n> --merge`) so the branch's individual commits — and their per-commit
history/provenance — are preserved on the target branch. Squashing collapses that history
and is not allowed.

## 10. Beaker / AI Hub Launch & Scheduling

- If you trigger jobs from Allen's HPC, use the `disrnn-cpu` conda env
  (`/allen/aind/scratch/han.hou/miniforge3/envs/disrnn-cpu`), not base. Treat this as the
  launcher/control-plane environment for `wandb`, `beaker`, and YAML tooling.
- **Submit ONLY to `hub` clusters** (the team's pools: `octo-hub-*`, `octo.hub-*`, `aihub-*`).
  **NEVER** to non-hub clusters (`aipbd-*`, `siti-*`, `dev-*`, other `octo.ai-*`) even if idle
  — they're not ours. Verified exceptions (non-hub `octo.ai-aws-*`, but admit our
  **low-priority preemptible** jobs — AWS, reach S3): `ai1/octo.ai-aws-g6e` (L40S bundle)
  and `ai1/octo.ai-aws-p5en` (H200 141 GB bundle).
- **Check schedulable capacity BEFORE launching any large job (> 4 GPUs / > 4 concurrent
  tasks): `python code/check_gpu_availability.py`** (Beaker + HPC). Route to whichever backend
  has room. "Free" is not "schedulable": Beaker reports GPUs on **cordoned** nodes as free
  (unschedulable), and `sinfo` counts `drain`/`down` nodes — the script strips both. Do not
  trust `beaker cluster list` / raw `sinfo` free counts, and never assume the cluster order
  below has open slots.
- Cluster choice: pick by **live schedulable capacity first**. `ai1/octo.ai-aws-g6e` (L40S,
  low/preemptible-only exception) and `ai1/octo-hub-aws-l40s` (L40S) for general jobs;
  `ai1/octo-hub-onprem-h200` / `ai1/octo.ai-aws-p5en` (the H200 preemptible exception)
  **only when a task needs the 141 GB** (wide `hidden_size=256`
  OOMs a 48 GB L40S). **H200 is not inherently faster than L40S/g6e** — do not prefer it on
  speed grounds. When all Beaker clusters are saturated/cordoned, HPC SLURM (`aind` partition)
  is the GPU overflow — check it with `--hpc` and launch there (§13 load-balancing).
- Heavy work never on the login node (see §5).
- **Three job-protection tiers in our workspace:** (1) **4 allocated** slots — protected
  non-preemptible (`{normal/high, preemptible: false}`), never evicted; (2) **8 unallocated**
  slots — normal-priority preemptible, hard cap; (3) **~unlimited low-preemptible**
  (`{low, preemptible: true}`) — bursts past the 8-cap onto spare GPUs, evicted first,
  auto-resumes. Default fan-outs → tier 3 (`priority: low`); must-finish runs → tier 1.
- Scheduling detail — the tier table/measurements, GPU-bundle sizing (`--memory 90GiB --cpu 12`
  = 1 L40s GPU), the g6e/p5en exceptions, cross-cloud S3 caveat, quota debugging, one-unit
  validation: the **`beaker-launch` skill** (`aind-behavior-foundation-model-skills/skills/beaker-launch/`,
  read before any non-trivial launch).

## 11. Verify Mechanisms With Data Before Asserting

When explaining *why* infra/scheduling/quota behaves a certain way, **pull the actual data
first** (`beaker ... --format json`, `cluster get`, the W&B API) and cite the field. Label
observed fact ("verified: …") vs inference ("likely, unconfirmed: …"); don't present a
hypothesis as a conclusion; isolate variables before attributing cause. Worked examples:
the `beaker-launch` skill, `references/scheduling-lessons.md`.

## 12. Post-hoc Analysis & Reporting

Reports are code: committed, regenerable, one producer per artifact. Every analysis JSON
carries a `_meta` block; every report file under `studies/<study>/analysis/reports/r*.md`
has YAML frontmatter (`id`, `status`, `wandb_groups`, `inputs`, `reproduce`) and uses
`<!-- BEGIN result-N -->` / `<!-- END result-N -->` markers around any region a script
regenerates. W&B pull caches are `.gitignore`'d. Full conventions — folder layout, file
contracts, `Makefile` convention, enforcement layers, multi-agent collaboration rules: the
**`posthoc-reporting` skill** (`aind-behavior-foundation-model-skills/skills/posthoc-reporting/`);
mirror a normalized study (e.g. `studies/01-gru-scaling-law/`).

## 13. Claude Science Workflow

The agent runs on the user's Mac (persistent brain); GitHub is the source of truth.
Two checkouts track it: the Mac clone (`~/Scripts/aind-disrnn-dispatcher`, authoring)
and the HPC login node (`/home/han.hou/code/...`, pull-only runtime). Load balancing:
CPU jobs → HPC SLURM, GPU jobs → Beaker — both launchers live here and only submit, so
one repo drives both. The sandbox cannot create a `.git` dir in a granted path, so the
user owns cloning (and SSO auth); the agent edits/commits/pushes into existing checkouts.
Full scheme — task-to-host table, W&B-from-sandbox access, credentials: the
**`codebase-map` skill**, `references/claude-science-workflow.md`.

- **Launching from the Claude Science Mac sandbox** (no HPC hop): the launchers
  (`code/launch_beaker.py`, `beaker_client.py`, `launch_beaker_resumable.py`) are
  sandbox-safe (W&B GraphQL + `Config(user_token=$BEAKER_TOKEN)` directly). Run from
  `code/` as `PYTHONPATH="$(pwd):$PYTHONPATH" python launch_beaker.py ... --no-submit`
  (`PYTHONSAFEPATH=1` in the sandbox breaks sibling imports otherwise; `--no-submit`
  is the safe dry-run). `beaker.org` + `api.wandb.ai` each need a one-time
  `request_network_access` grant. Full recipe: the `beaker-launch` skill,
  `references/sandbox-launch.md`.
- **Verify the image name before submitting.** Old example specs
  (`experiment_h100/h200/pack.yaml`) reference `beaker: han-hou/disrnn-wrapper`,
  which no longer exists (→ `ImageNotFound`/404). Current:
  `han-hou/disrnn-wrapper-main-20260712` (see `code/beaker/README.md` "Available
  images"); list live images with
  `beaker workspace images ai1/aind-dynamic-foraging-foundation-model`. Code is
  pulled fresh at startup (`entrypoint.sh` checks out `WRAPPER_REF`/`DISPATCHER_REF`/
  `FORAGING_MODELS_REF`), so code/config edits need **no** rebuild — only a stale
  image or changed dependencies do.
- **Transient node failure ≠ code bug.** A job dying in ~5 s with
  `status.message: "no space left on device"` / `started=None` is a full-NVMe node;
  resubmit (lands elsewhere) instead of debugging training code.
