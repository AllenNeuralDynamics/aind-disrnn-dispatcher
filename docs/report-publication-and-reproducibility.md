---
aliases:
  - publication layers
  - reproducibility plan
  - docs website
  - CO analysis capsule
tags:
  - planning
  - reproducibility
status: proposed
---

# Report publication & reproducibility layers

> **Status:** proposed 2026-07-16, alongside the approved repo split
> ([[repo-split-plan]]). All layers live in the future `aind-disrnn-studies`
> repo and sequence *after* the split.

## Priorities (in order)

1. **Reproducibility** — anyone (human or agent) regenerates every figure and
   report block from the repo alone, offline, deterministically.
2. **AI-native** — the contract is plain files + `make` + git diffs, so an
   agent can regenerate, verify, and review without platform accounts or UIs.
3. **Always-visible reports** — browsable at all times (GitHub today; a docs
   site as a nicer front door).
4. **Code Ocean** — optional last-mile packaging of the same contract for
   institutional sharing. **Never a priority**; it must fall out for free
   from layers 1–2, not drive them.

## The load-bearing fact (verified 2026-07-16)

Studies 03, 04, and 05 are **already offline-reproducible by convention**
(the posthoc-reporting "freeze the numbers" rule). Their Makefiles separate
two steps:

- **Pull** (e.g. `make pull`, `analysis/pull_grid.py`) — reads W&B, writes a
  *committed curated grid CSV*. Needs `WANDB_API_KEY` + network; run on an
  authenticated node. This is the only step that touches live W&B.
- **Produce** (`make all`) — reads only committed inputs, regenerates
  figures + `BEGIN/END` report blocks. Offline, deterministic.

Each Makefile header states it: *"the committed CSV is the default source of
truth, so `make` runs offline."*

**Gaps** (the whole layer-1 work list):

- **Study 01**: five of the seven producers in `make all`
  (`bootstrap_scaling.py`, `build_report.py`, `generative_match.py`,
  `nxd_scaling.py`, `rl_baseline.py`) open `wandb.Api()` directly — it
  predates the convention. (`update_final_report_nxd.py` and `update_r8.py`
  already read committed JSONs only.)
- **Study 05, r4** (`generative_report.py` path): reads W&B rollout
  histories directly instead of a committed input.

An earlier idea — a `DISRNN_WANDB_CACHE` env var with the raw W&B pull cache
attached as a CO data asset — is **rejected**: committed curated grids are
strictly better (reviewable diffs, no side-channel state, no data-asset
plumbing), and they are already the house pattern.

## Layer 1 — finish "freeze the numbers" (reproducibility)

Normalize studies 01 and 05-r4 to the pull/produce split: add pull producers
that write committed curated CSVs; make report producers read them. After
this, `for s in studies/0*/; do make -C "$s" all; done` succeeds offline from
a fresh clone. This single property is what every later layer runs on.

-> verify: `make all` in all five studies exits 0 with `WANDB_API_KEY` unset
   and network to `api.wandb.ai` blocked.

## Layer 2 — CI regeneration check (AI-native, always-on)

GitHub Actions on PR + push to `main` in the studies repo: run every study's
`make all`, then fail if `git status` is dirty beyond allowed churn
(`_meta.produced_at_pt`, `*_git_sha` stamps). This guarantees committed
reports never drift from their producers, and gives agents a machine-checkable
contract ("my regeneration matches what's committed") instead of a manual
convention.

Cheap because of layer 1: CPU-only, no secrets in CI.

## Layer 3 — docs website (always-visible reports)

MkDocs Material + GitHub Pages, built by the same CI **on push to `main`**
(not per release — the repos cut no release tags, reports iterate daily, and
the merge-commit-only PR policy already makes merge-to-main the meaningful
gate). Nav auto-generated from `studies/*/analysis/reports/r*.md` +
per-study READMEs; figures come from the committed PNGs (see the
committed-artifacts policy in [[repo-split-plan]]); the frontmatter `status:`
field badges draft vs final. If release discipline appears later, `mike` adds
versioned docs without restructuring.

GitHub browsing keeps working regardless; the site is a front door, not a
replacement.

## Layer 4 — CO analysis capsule (optional, last)

**Scope guard: analysis regeneration only.** This is a *separate* capsule
from the dispatcher's existing launch capsule (whose app panel submits Beaker
jobs). The analysis capsule never triggers Beaker/HPC/training — its
Reproducible Run is exactly:

```text
clone aind-disrnn-studies @ <pinned SHA>
for s in studies/0*/; do make -C "$s" all; done
copy regenerated reports + figures to /results
```

Because of layer 1 this needs **no W&B secret, no network, no data asset** —
the button is hermetic by construction, and "another user presses
Reproducible Run and all figures regenerate" holds forever. Environment:
CPU-only (python + pandas/matplotlib/etc. per the studies' analysis deps);
no JAX/GPU, since `make all` producers read committed grids, not models.

Build it at the first study freeze, or whenever institutionally useful.
If layer 1 is done, this is an afternoon; if it ever isn't an afternoon,
that signals layer 1 has regressed — fix that instead.

## Sequencing

1. Repo split ([[repo-split-plan]], approved).
2. Layer 1 (study 01 + 05-r4 normalization) — the only real code work.
3. Layer 2 + 3 together (one CI workflow: check, then publish site).
4. Layer 4 when a study freezes. Never blocks anything.

## Related

- [[repo-split-plan]] — the split this builds on; committed-artifacts policy;
  the CO "launch surface and archive, not a home" decision this refines.
- [[posthoc-analysis]] — the "freeze the numbers" / curated-grid convention
  that layer 1 finishes rolling out.
- [[study-organization]] — study self-containment that makes per-study
  `make all` the natural unit.
