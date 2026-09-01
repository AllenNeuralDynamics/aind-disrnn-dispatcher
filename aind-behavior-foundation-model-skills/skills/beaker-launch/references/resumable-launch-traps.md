# Resumable-launcher traps — mechanism and evidence

Three failure modes of `code/launch_beaker_resumable.py`. The one-line rules live in
`SKILL.md`; this file is why each exists, so the rules can stay short without becoming
folklore.

## 1. Resolved-JSON payload ceiling (~15 tasks)

Beaker rejects a single-experiment payload above roughly 15 tasks with a **409 whose
message does not mention size** — it reads like a name conflict, which sends you looking
in the wrong place. The launcher expands a grid into one self-contained task per grid
point, each carrying its full env and command, so the resolved JSON grows linearly with
the grid and crosses the limit sooner than task count alone suggests.

Procedure: render with `--no-submit`, look at the resolved-JSON size, and if it is too
big, split the grid into ≤ ~15-task chunks and submit each directly:

```bash
beaker experiment create -w "$WS" <spec>.yaml
```

## 2. `wandb.project` defaults to `test`

Two launchers, two different owners of the W&B project:

| Launcher | Who sets the project |
|---|---|
| `launch_beaker.py` (native) | the `wandb agent` sweep controller, from the sweep YAML's top-level `project:` |
| `launch_beaker_resumable.py` | **nobody** — each task runs `run_hpc` directly, so the project comes from Hydra's `wandb.project`, which defaults to `test` |

The resumable launcher sets the W&B **group** (`<variant>@<launch_id>`) but not the
project. The sweep's top-level `project:` field is read only by the native controller, so
it looks correct in the YAML while every run lands in `test`. Nothing errors; the runs
simply are not where the study's analysis looks for them.

Fix: add to the sweep's `command:` list, next to `wandb.tags`:

```yaml
command:
  - wandb.project=<study_project>
  - wandb.entity=AIND-disRNN
```

## 3. Branch refs break resumes (pin a SHA)

A preempted task auto-resumes by re-running `entrypoint.sh`, which re-checks-out the ref
it was given. A branch ref is mutable, so a resume can:

1. **Fail** — the branch no longer exists. GitHub auto-deletes branches on PR merge, so
   merging your own fix branch while the grid is running breaks every subsequent resume.
2. **Silently run different code** — the branch advanced, so the second half of a
   preempted run executes the new tip. The run's own first half and second half no longer
   agree, and nothing in the record says so.

A full SHA is immutable: the resume re-checks-out the *same* commit, always resolvable.
The launch record logs a commit for provenance, but that records *intent* — only pinning
the ref itself makes the *executed* code match the record. The entrypoint already accepts
a SHA (`git fetch --depth 1 origin <sha>`).

Resolve against the **remote**, since the container fetches from origin — a
local-only commit is not reachable:

```bash
git ls-remote origin <branch> | cut -f1   # -> WRAPPER_REF / DISPATCHER_REF
```

Branch refs are fine only for one-shot, non-resumable jobs, which have no resume window.

If this keeps biting, the durable fix is to teach the launcher to resolve branch→SHA at
submit time: you keep typing branch names, and the task spec pins the SHA.
