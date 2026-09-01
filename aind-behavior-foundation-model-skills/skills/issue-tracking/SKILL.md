---
name: issue-tracking
description: File GitHub issues for the AIND behavior foundation model repos and put them on the AIND-behavior-fm org project board (project 184) with Status, Priority, and Size set. Covers the always-file-an-issue-first rule, the cached board field/option IDs, the REST+GraphQL calls (no gh CLI in the sandbox), how to pick P0/P1/P2 and XS-XL, and linking the PR back. Use before starting any tracked work in aind-disrnn-dispatcher or aind-disrnn-wrapper, whenever opening an issue or PR, or when a board field needs updating.
---

# Issue & project-board tracking

**Every piece of tracked work starts with an issue, and every issue goes on the board with
Status, Priority, and Size set.** An issue nobody can find on the board is invisible at
planning time, and a PR with no issue leaves the *why* only in a diff. File first, then work.

Board: <https://github.com/orgs/AllenNeuralDynamics/projects/184/> ("AIND-behavior-fm").
Issues live on the repo the work touches — usually `aind-disrnn-dispatcher`, or
`aind-disrnn-wrapper` when the change is to the training/analysis payload. A two-repo
change gets one issue on the repo that owns the *decision*, with the second repo's PR
cross-linked from it.

## The four steps

```bash
python scripts/board.py --repo aind-disrnn-dispatcher \
    --title "..." --body-file issue.md \
    --labels documentation priority:P1 \
    --status "In progress" --priority P1 --size M
```

That script does all four steps and prints the issue URL. Do them by hand only if you need
something it does not cover — the calls are below.

1. **Create the issue** (REST `POST /repos/{org}/{repo}/issues`) with `labels`.
2. **Add it to the board** (GraphQL `addProjectV2ItemById`) — returns the *item* id, which
   is what the field mutations take. An issue node id is not an item id.
3. **Set Status, Priority, Size** (GraphQL `updateProjectV2ItemFieldValue`, once per field,
   `value: {singleSelectOptionId: ...}`).
4. **Read the item back** and confirm all three fields landed. The mutations return success
   for a valid-but-wrong option id, so a read-back is the only proof.

## Choosing the field values

**Status** — where the work actually is, not where you hope it goes:
`Backlog` (filed, not scheduled) → `Ready` (scoped, someone can pick it up) →
`In progress` (being worked *now*) → `In review` (PR open) → `Done` (merged).
Set `In progress` when you file an issue you are about to start, and flip to `In review`
in the same session you open the PR.

**Priority** — what breaks if it waits:

| | Use for |
|---|---|
| `P0` | Blocks a run, a deliverable, or the manuscript path; or produces wrong published numbers |
| `P1` | Actively misleads or wastes time (stale guidance, a footgun, a missing guardrail) but nothing is blocked today |
| `P2` | Real improvement, no active harm — cleanups, nice-to-haves, deferred refactors |

**Size** — implementation effort, *not* importance. `XS` one-line/config; `S` one file or a
contained change; `M` several files or one subsystem, still one PR; `L` multi-PR or spans
both repos; `XL` needs decomposing into child issues before it can start.

## Issue body

Enough for someone with no context — including you in three months — to act:
**Context** (what prompted this) → **Root cause** if known, not just symptoms →
**Findings** (a table when there are several) → **Done when** (a checkbox list that is
literally the acceptance criteria) → **Notes** (constraints, gotchas, links).

State what you *verified* versus what you *suspect*, per AGENTS.md §11. A finding with a
file:line reference is actionable; "the docs seem stale" is not.

## Linking the PR

Put `Closes #<n>` in the PR body so the merge closes the issue, move the board item to
`In review`, and let the merge move it to `Done`. When a PR only partly addresses an
issue, write `Refs #<n>` instead and leave the issue open — an auto-closed issue with
unfinished "Done when" boxes is worse than an open one.

Labels available on the dispatcher: `bug`, `documentation`, `enhancement`, `evaluation`,
`extension`, `interpretability`, `baselines`, `training/pipeline`, `validation`, `blocked`,
`ready-for-agent`, `rename-migration`, `question`, `priority:P0/P1/P2`. The `priority:*`
label duplicates the board field on purpose — the label is visible in issue lists where
board fields are not.

## Sandbox / auth notes

- **No `gh` CLI** in the Claude Science sandbox. Use REST + GraphQL directly with
  `GITHUB_TOKEN` (`urllib`/`requests` + `Authorization: Bearer`).
- **`AllenNeuralDynamics` enforces SAML SSO.** A classic PAT must be SSO-authorized for the
  org or every call returns 403 "Resource protected by organization SAML enforcement". This
  is a one-time authorization in the token's settings, not something to retry around.
- Setting a board field needs the `project` scope; creating issues needs `repo`.

## Cached IDs (verify before trusting)

Field and option ids are stable until someone recreates a field, so treat these as a cache
that saves a round-trip, not as truth. `scripts/board.py --discover` re-reads them; the
GraphQL query is in `references/board-api.md`.

| | ID |
|---|---|
| Project `AIND-behavior-fm` (#184) | `PVT_kwDOBa47bs4BIeG5` |
| Status field | `PVTSSF_lADOBa47bs4BIeG5zg4672s` |
| — Backlog / Ready / In progress / In review / Done | `f75ad846` / `61e4505c` / `47fc9ee4` / `df73e18b` / `98236657` |
| Priority field | `PVTSSF_lADOBa47bs4BIeG5zg467-8` |
| — P0 / P1 / P2 | `79628723` / `0a877460` / `da944a9c` |
| Size field | `PVTSSF_lADOBa47bs4BIeG5zg467_A` |
| — XS / S / M / L / XL | `6c6483d2` / `f784b110` / `7515a9f1` / `817d0097` / `db339eb2` |

The board also carries `Estimate` (number), `Start date`, `Target date`, `Parent issue`, and
`Sub-issues progress` — leave them unset unless the work is being scheduled.

## Reference

- `references/board-api.md` — the raw discovery query and each mutation, for when the
  script does not fit (bulk moves, sub-issues, date fields).
