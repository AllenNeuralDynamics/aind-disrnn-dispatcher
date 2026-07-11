---
name: study-conventions
description: Organize studies, variants, and launch provenance in this repo — studies/<study>/variants/<variant>/ layout, one W&B project per study, group naming <variant>@<launch_id>, and the meta.* provenance block. Use when creating a new study or variant, deciding where a launch's configs live, or naming/interpreting W&B groups.
---

# Study & experiment organization

Canonical detail: `docs/study-organization.md`. **If this skill and the doc conflict,
the doc wins.**

## Core scheme

- A **study answers one scientific question**; its conditions are **variants**, not new
  studies. New top-level `studies/<name>/` only for a genuinely different question
  (different model family, metric, or task).
- **Study folder name = `NN-{model}-{purpose}`**, kebab-case, no dates. `NN` is a
  zero-padded 2-digit **accession number** (stable creation-order id, assigned at
  creation from the next free number — NOT an execution order; never renumber or
  reuse). `{model}` is the model family (`gru`, `disrnn`; use `gru-vs-disrnn` for a
  cross-model comparison study). `{purpose}` is the question
  (`scaling-law`, `scaling-law-ignore`, `beta-scan`). Examples: `01-gru-scaling-law`,
  `02-gru-scaling-law-ignore`, `03-disrnn-beta-scan`. The folder name is independent
  of the W&B **project** name (e.g. folder `02-gru-scaling-law-ignore` ↔ project
  `mice_ignore_scaling`) and of the immutable `meta.study` stamped on already-logged
  runs — when renaming an existing study, note `formerly: <old-name>` in its README so
  the folder ↔ run-stamp mapping stays explicit, and leave historical launch records
  (which record the path used at launch time) unrewritten.
- **One folder per study**: `studies/<study>/` — shared analysis scripts, reusable
  configs, and a README with a **Variants index** table (what differs, status, W&B
  group, Beaker experiment id) at the study root.
- **Variants as self-contained subfolders**: `studies/<study>/variants/<variant>/` with
  its `sweep.yaml`, `experiment.yaml`, `notes.md` (what differs + result + W&B group +
  Beaker exp id), and launch records. Name descriptively (`v2-postwarmup`,
  `hsize-scan`), never by date.
- **One W&B project per study, one group per variant** (group set via the sweep's
  `name:`) — keeps variants side-by-side comparable; never a project per variant.

## Provenance (one launch == one "pseudo-sweep")

- **W&B group = `<variant>@<launch_id>`**, launch_id = Seattle timestamp. The
  launch_id is also folded into run ids so repeated launches never collide.
- **`meta.{study,variant,launch_id,label,note,config_hash}`** — portable across
  CO / Beaker / Allen HPC. `launch_beaker_resumable.py` derives study/variant from the
  `studies/<study>/variants/<variant>/` path and injects `DISRNN_META_*` env; the
  wrapper's `start_wandb_run` stamps it onto the run.
- **Always pass `--note`** ("why this run exists + what we want to learn") so the
  scientific intent is readable straight from the W&B record.
- Platform-native ids are stamped alongside: `BEAKER_EXPERIMENT_ID`, `BEAKER_JOB_ID`,
  `CO_COMPUTATION_ID`, plus `wrapper_commit` / `dispatcher_commit`. Both Beaker
  launchers (`launch_beaker_resumable.py` and `launch_beaker.py`) implement this
  identically.

## Checklist for a new variant launch

1. Create `studies/<study>/variants/<variant>/` with `sweep.yaml` + `experiment.yaml`
   (copy the closest existing variant).
2. Write `notes.md`: what differs from the sibling variants and what you expect.
3. Launch via the beaker-launch or hpc-launch skill with `--label` and `--note`.
4. Add a row to the study README's Variants index.
5. After the group settles, write `launch_record_<label>/results.md`
   (contract in the posthoc-reporting skill).

## Housekeeping: wrapping up a completed study

When a study's experiments are done and results are settled: normalize the folder,
clean up runs, and get the work onto the integration branch. Verified end-to-end
procedure.

### 1. Normalize the study folder
Mirror an already-normalized reference study (e.g. `studies/data-scaling-law/`,
`studies/ignore-trials-scaling/`). Target layout:
- `analysis/` — producer script(s) (pull live W&B → curated `*.json` with a `_meta`
  block + `*.csv` + key figure), `wandb_keys.py` (comment-only contract naming the
  exact summary keys read), `reports/` (`rN-*.md` + `INDEX.md`), `provenance/`.
- Study root: `Makefile` (one PHONY target per report + `all`), `environment.lock`
  (pin the wrapper commit that PRODUCED the metrics — `_meta.build_meta` parses it),
  `CHANGELOG.md`, `README.md` with a **Verdict** section (the scientific findings).
- **Reuse shared helpers in `studies/util/`** — `_meta.py` (provenance `_meta` block;
  pass `study_root=`) and `plot_style.py` (presentation rcParams + palette). Don't
  copy them per-study.
- **Commit derived outputs** (JSON/CSV/PNG) so results render directly in the repo;
  gitignore only caches + `__pycache__`. Goal: reports human-readable, provenance
  AI-readable but human-auditable.
- Reports use script-owned `<!-- BEGIN result-N -->` / `<!-- END result-N -->` blocks
  (regenerated by the producer) with human Discussion prose OUTSIDE the markers; the
  pipeline must be idempotent (re-run → empty diff). Full contract in the
  posthoc-reporting skill.

### 2. Clean launch records
Tuck loose `sweep_*.yaml` launch specs under
`variants/<variant>/launch_record/_sweeps/`, write per-variant
`launch_record/results.md`, and Trash superseded root-level scratch. In shared
working trees, leave files that belong to a concurrent study/session alone.

### 3. Clean stale W&B runs (IRREVERSIBLE — confirm first)
Enumerate every run in the study's W&B project, classify by state, and delete
**only** failed/crashed runs that carry **no usable metrics**. **Never delete
main / restored / follow-up runs.** Verify no deletion target holds unique data,
then show the classified keep/delete list and get explicit confirmation before
deleting. (GraphQL `deleteRun` payload has `clientMutationId` + `numDeleted`, not
`success` — selecting `success` returns HTTP 400.)

### 4. PR to the integration branch
Open the study PR toward the integration branch (`ai_hub_pck_integration`). If the
study branch has commits from another study interleaved, build a **clean squash**
directly off the target base containing only this study's + shared files (drop the
other study's paths; revert any of its config default-flips). Merge dependency
order: **wrapper PR first** (its commit is pinned in `environment.lock`), then
dispatcher. Update the relevant roadmap issue with the verdict + PR links.

### 5. Retire the branch (only after content is captured)
Verify the branch is fully captured with a **two-dot** content diff
(`git diff --name-only <base> <branch>` on the study/shared paths — zero diff =
captured), **NOT** GitHub's three-dot compare (diffs from the merge-base, always
shows divergence after a squash). Any sibling docs commit merged ONTO the branch
after the squash must be PR'd up separately first, or it orphans. `git branch -d`
refuses a squash-merged branch — use `-D` once the two-dot diff confirms capture.
Record the tip SHA before deleting (recoverable). Branches with unmerged unique
commits, or another active session's work, are off-limits.
