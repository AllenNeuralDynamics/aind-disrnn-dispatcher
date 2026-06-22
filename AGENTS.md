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

A study answers one scientific question; its many runs/conditions are *variants* of that
question, not separate studies.

- **One folder per study** under `studies/<study-name>/`. Shared tooling lives at the study
  root (analysis scripts, reusable configs, README).
- **Variants as subfolders:** `studies/<study>/variants/<variant-name>/`, each self-contained
  — its `sweep.yaml`, `experiment.yaml`, a `notes.md` (what differs + result + W&B group +
  Beaker exp id), and its launch record. Name variants descriptively (`v2-postwarmup`,
  `hsize-scan`), not by date.
- **One W&B project per study, one group per variant** (set the group via the sweep's
  `name:`). This keeps every variant directly comparable side-by-side in a single project —
  prefer this over a project-per-variant.
- The study README carries a **Variants index** table (one row per variant: what differs,
  status, W&B group, experiment id).
- Spin up a **new** top-level `studies/<name>/` only for a genuinely different question
  (different model family, metric, or task) — not for a variant of the same one.

**Provenance / tracking (one launch == one "pseudo-sweep").** Every launch is uniquely
and *readably* identifiable, with platform-native ids saved alongside for cross-ref:
- **W&B group = `<variant>@<launch_id>`** (launch_id = Seattle timestamp). Distinguishes
  repeats of a variant; readable (variant → study folder, launch_id → time). `launch_id`
  is also folded into run ids, so repeats get unique ids (and the deleted-id resync trap
  is avoided).
- **`meta.{study,variant,launch_id,label,note,config_hash}`** — our portable system,
  consistent across CO / Beaker / AI1 HPC. Set by `launch_beaker_resumable.py` (derives
  study/variant from the `studies/<study>/variants/<variant>/` path) via `DISRNN_META_*`
  env; stamped by the wrapper's `start_wandb_run`. **`note`** is free-text "why this run
  exists + what we want to learn", injected by either launcher's `--note` so humans and
  agents can read a run's scientific intent straight from the W&B record (no second lookup).
- **Platform-native ids saved next to `CO_COMPUTATION_ID`**: `BEAKER_EXPERIMENT_ID`,
  `BEAKER_JOB_ID` (read from Beaker env by the wrapper — route-agnostic, so this works for
  both the resumable launcher and the native `wandb agent` route), plus `wrapper_commit` /
  `dispatcher_commit`.
- **Both launchers** implement this identically: `launch_beaker_resumable.py` (pseudo-sweep)
  and `launch_beaker.py` (native `wandb agent` sweep) share the helpers and both inject
  `WANDB_RUN_GROUP` + `DISRNN_META_*`. The native route additionally has a real W&B sweep
  as its platform-native launch id; the wrapper stamps Beaker/CO ids for both routes.

## 9. Merging Pull Requests

**Never squash-merge a PR.** Merge with a **merge commit** (no fast-forward, e.g.
`gh pr merge <n> --merge`) so the branch's individual commits — and their per-commit
history/provenance — are preserved on the target branch. Squashing collapses that history
and is not allowed.
