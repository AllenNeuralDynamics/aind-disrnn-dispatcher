---
aliases:
  - repo split plan
  - split studies
  - aind-disrnn-studies
tags:
  - planning
  - migration
  - one-shot
status: proposed
---

# Repo-split plan: extract `studies/` into `aind-disrnn-studies`

> **Status:** proposed, not executed. Delegated to a separate agent for execution.
> Written after the `posthoc-analysis` standard-structure migration on the
> integration line, ending at commit `122abfd` and later merged to `main`.

## TL;DR

Split `aind-disrnn-dispatcher` at the framework/application seam:

- **`aind-disrnn-dispatcher`** (this repo, stays) — launchers (`code/`),
  framework docs (`docs/`), Docker env, CO capsule metadata, root `AGENTS.md`.
- **`aind-disrnn-studies`** (new sibling repo) — everything currently under
  `studies/`, one study per subfolder, each self-contained per
  `docs/study-organization.md`.

Pilot: `studies/data-scaling-law/` (the only study today and the most
standard-conformant one). Preserve per-file history via `git filter-repo`.

## Should we do this now? (discussion 2026-06-30)

**Current lean: not yet.** The split is a defensible long-term direction but
premature today, and separating the *analysis* layer — not the whole `studies/`
tree — is the sharper cut. Reasoning, from a 2-people-plus-AI project's point of
view (don't over-engineer):

- **Only one study exists.** Standing up a second repo to organize a single
  folder pays multi-repo overhead (side-by-side clones, cross-repo PRs,
  container entrypoint changes, a 3-way version matrix for provenance) with no
  offsetting benefit yet. Revisit when there is a *second study* or a *second
  contributor who only touches studies* — either gives the seam a concrete job.
- **Package before you split.** The launch-side coupling
  (`studies/*/variants/` → `code/` launchers) is a *shell path* (`python
  ../aind-disrnn-dispatcher/code/launch_*.py`), not a real interface. That is
  fragile in containers and unversioned. The clean boundary only exists once
  `code/` is a pip-installable package the studies repo depends on with a pinned
  version — exactly how the study already consumes the wrapper via
  `environment.lock`. Splitting first leaves you in the fragile relative-path
  phase indefinitely. So: **package `code/` is a prerequisite, not a non-goal**
  (supersedes the "Non-goals" bullet below for sequencing purposes).

### The analysis layer is the most-separable piece — and it is already split

The "analysis" concept is *already* bisected across the two existing repos, along
a natural axis, with **W&B as the boundary**:

- **Producer (per-run, in the wrapper):**
  `aind-disrnn-wrapper/code/post_training_analysis/` (`generative_analysis.py`,
  `heldout_finetuning.py`, `likelihood_*`, `baseline_rl_analysis.py`,
  `embedding_space_analysis.py`), invoked in the capsule via
  `run_analysis.py <subcommand>` (generative / from-histories /
  likelihood-comparison / likelihood-advantage / embedding / baseline-rl /
  finetune). Needs the trained model + data + JAX/GPU. Writes quantitative
  summaries into each run's **W&B summary + logged artifacts**.
- **Consumer (cross-run, in the study):** `studies/data-scaling-law/analysis/`
  reads those summary keys/artifacts back via `wandb.Api()`, aggregates across
  cells (D × seed × subject), fits curves, renders `reports/r*.md`. It does
  **not** import `post_training_analysis`; verified it has **zero references to
  `code/`** either. Its only inbound contract is a handful of W&B key strings.

So the study `analysis/` is the single most extractable component (its input is
the cloud, keyed by group name). But it is also the most *semantically* bound to
its study — it **is** the study's answer — so pulling it into its own repo fights
the "a study is self-contained" principle in `study-organization.md`. Net: leave
it in place; the fragility to fix is the *contract*, not the location.

### The wrapper↔study analysis contract (measured, then hardened in place)

Measured 2026-06-30: the consumer hardcodes **7 distinct W&B summary keys**; the
key *vocabulary* has been stable in wrapper history (churn is in analysis
internals, not key names). The real hazard was the **failure mode**: reads used
`summary.get(key)` then `if None: continue`, so a renamed/dropped key **silently
drops runs** — a report would shrink and its numbers shift with no error.

Right-sized fixes applied (no shared schema package, no CI validator, no
machine-readable manifest — those exceed the benefit at this scale):

1. **`analysis/wandb_keys.py`** — single source of truth for the 7 keys
   (builders + constants) plus a `require()` helper. The whole wrapper-contract
   surface is now one greppable file to review when bumping the wrapper pin.
2. **Loud-on-schema-break guards** — `generative_match.py` and `nxd_scaling.py`
   still skip individual partial runs, but now `raise KeyError` if *all* cells
   lack the required key (the rename signal), pointing at `wandb_keys.py`.
3. **`_meta.wrapper_git_sha`** — every analysis JSON now stamps the wrapper
   commit (read from `environment.lock`) that produced the keys, alongside
   `dispatcher_git_sha`. Closes the producer-side provenance gap.

`analysis/watch_nxd_d30.py` (a defunct one-off watcher that only presence-checks
readiness, no report corruption risk) was intentionally left untouched.

Because that contract is small (7 keys) and stable, it is genuinely fine to keep
the analysis layer where it is; these guards make the W&B boundary safe to cross
whether or not analysis ever becomes its own repo.

## Prerequisites (do these before executing)

1. **All in-flight PRs against dispatcher must be merged first.** The integration
   line is now merged to `main`; any commits not on `main` at split time will be
   silently dropped from the studies extract.
   -> verify: `git log origin/main..HEAD -- studies/` prints nothing.
2. **Snapshot AGENTS.md rev**, since content flows across both repos.
   -> verify: record `git log -1 --format=%H AGENTS.md` in the studies-repo
   commit message.
3. **Confirm no import-time coupling** from `studies/**` onto `code/**`.
   -> verify (already true 2026-06-25): `rg -n "from (launchers|code|dispatcher)" studies/` is empty.

## Target state

### `aind-disrnn-dispatcher/` (framework, unchanged root)

```text
aind-disrnn-dispatcher/
├── AGENTS.md                    # framework behaviour rules
├── README.md                    # pointer to aind-disrnn-studies in "Studies" section
├── docs/                        # posthoc-analysis, study-organization, beaker-playbook
├── code/                        # launchers (launch_beaker*.py, launch_hpc.py, ...)
│   ├── beaker/                  # shared beaker templates
│   ├── config/                  # shared configs consumed by launchers at runtime
│   ├── hpc/                     # shared SLURM helpers
│   └── util.py
├── environment/                 # Dockerfile
├── .codeocean/                  # CO capsule metadata
└── (no studies/)
```

### `aind-disrnn-studies/` (new sibling repo)

```text
aind-disrnn-studies/
├── AGENTS.md                    # studies-specific rules; Related: back to dispatcher AGENTS
├── README.md                    # index of studies; how to clone alongside dispatcher
├── data-scaling-law/            # (was studies/data-scaling-law/, promoted to root)
│   ├── analysis/
│   ├── variants/
│   ├── reports/
│   ├── Makefile
│   ├── environment.lock
│   ├── CHANGELOG.md
│   └── README.md
└── .gitignore                   # ignore per-study analysis/_cache_*.json, etc.
```

Alternative: keep the `studies/<name>/` wrapper (i.e. mirror the current
prefix). Trade-off:
- **Flat** (`data-scaling-law/` at root): tighter for a repo whose sole
  purpose is studies; matches "one folder per study" cleanly.
- **Prefixed** (`studies/data-scaling-law/`): zero-diff paths for docs,
  Makefile, tools that hardcode the `studies/` prefix; easier `filter-repo`.
  Recommended for the first split; can flatten later.

**Decision needed** — see Open Questions Q1. This plan assumes **prefixed**
below to minimise churn.

## What moves, what stays

| Path | Action |
|---|---|
| `studies/**` (tracked files only) | -> `aind-disrnn-studies/studies/**` |
| `code/**` | stays in dispatcher |
| `docs/**` | stays in dispatcher; studies repo links back |
| `environment/**` | stays in dispatcher |
| `.codeocean/**` | stays in dispatcher |
| `AGENTS.md` | stays; studies repo gets a **new** `AGENTS.md` that inherits via `Related` link |
| `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md` | copy-forward (fresh, not filter-repo'd) into studies repo |
| `.gitignore` | copy-forward the studies-relevant rules; drop dispatcher-only rules |
| `artifacts/` | untracked already; regenerated locally in either repo |

## Cross-repo runtime dependency

After the split, study launchers still invoke the dispatcher's launchers by
shell. The conventional layout the two READMEs assume is **side-by-side clones**:

```text
~/code/
├── aind-disrnn-dispatcher/
└── aind-disrnn-studies/
```

Study docs are updated so that:

```text
# before (single repo, from repo root)
python code/launch_beaker_resumable.py ...

# after (from studies repo root, dispatcher cloned as sibling)
python ../aind-disrnn-dispatcher/code/launch_beaker_resumable.py ...
```

Ditto for content references (`code/config/model/gru_scaling.yaml` becomes
`../aind-disrnn-dispatcher/code/config/model/gru_scaling.yaml`).

Files to update in the studies repo after extraction (search for
`code/launch|code/config|code/hpc|code/beaker`):

- `studies/data-scaling-law/README.md`
- `studies/data-scaling-law/variants/*/notes.md`
- `studies/data-scaling-law/variants/*/sweep.yaml` (comment lines only)
- `studies/data-scaling-law/variants/*/launch_record/*.yaml` (comment lines only)
- `studies/data-scaling-law/launch_heldout_rerun.py` (docstring line 14)
- `studies/data-scaling-law/analysis/rl_baseline_verdict.md`
- `studies/data-scaling-law/analysis/rl_baseline.py` (line 486 error message string)

Long-term (out of scope for this split): promote the launchers to a proper
Python package with console-script entrypoints so the shell path drops out
entirely. Track as follow-up.

## Migration mechanics

### Step 1 — Extract `studies/` with full history

Run in a scratch clone (never on the working repo):

```bash
mkdir -p /scratch/repo-split && cd /scratch/repo-split
git clone --no-local https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher.git extract
cd extract
pip install git-filter-repo   # if not already installed
git filter-repo --path studies/
```

Result: `extract/` now contains only files under `studies/`, with per-file
history preserved (`git log --follow studies/data-scaling-law/analysis/rl_baseline.py`
should show every commit that touched it).

-> verify: `git log --oneline | wc -l` in `extract/` is > 20 (matches the
   number of dispatcher commits that touched `studies/`).
-> verify: `git log --follow --oneline studies/data-scaling-law/analysis/rl_baseline.py`
   shows at least commits `1e30716`, `6b0ecd5`, `122abfd`, etc.

### Step 2 — Add fresh top-level files

In `extract/`, add:

- `AGENTS.md` (studies-specific; see template below).
- `README.md` (index + side-by-side clone instructions).
- `LICENSE` (copy from dispatcher).
- `CODE_OF_CONDUCT.md` (copy from dispatcher).
- `.gitignore` (copy dispatcher's, prune dispatcher-only rules).

Commit as **one commit** with message:

```text
chore: initialise aind-disrnn-studies from aind-disrnn-dispatcher

Extracted studies/ from aind-disrnn-dispatcher via git filter-repo.
Source revision: <dispatcher HEAD sha at split time>
```

### Step 3 — Create the GitHub repo and push

```bash
gh repo create AllenNeuralDynamics/aind-disrnn-studies --public --confirm
git remote add origin git@github.com:AllenNeuralDynamics/aind-disrnn-studies.git
git push -u origin main
```

-> verify: `gh repo view AllenNeuralDynamics/aind-disrnn-studies` succeeds.

### Step 4 — Update the new repo's cross-repo references

Sweep the files listed under "Cross-repo runtime dependency" above; prefix
`code/...` -> `../aind-disrnn-dispatcher/code/...`. Commit:

```text
docs: point launcher references at sibling aind-disrnn-dispatcher clone
```

-> verify: `rg -n "python code/launch" studies/` returns no matches.
-> verify: dry-run a launch (`python -m py_compile` on the launch scripts;
   optionally trigger one `launch_beaker_resumable.py --help` from the
   studies repo to confirm the sibling clone assumption works).

### Step 5 — Remove `studies/` from dispatcher

On a fresh branch in the dispatcher working tree:

```bash
git checkout -b chore/remove-studies-after-split
git rm -r studies/
```

Update dispatcher `README.md` "Studies" section to link to
`aind-disrnn-studies`. Update `AGENTS.md` to remove study-specific rules
that migrate (if any — most are framework-general and stay).

Commit:

```text
chore(dispatcher): remove studies/ after extraction to aind-disrnn-studies

Studies extracted with full history to
https://github.com/AllenNeuralDynamics/aind-disrnn-studies (see that
repo's initial commit for the split source revision).
```

Open a PR; **merge with merge commit** (not squash) per AGENTS.md §9.

-> verify: dispatcher's CI still passes (if any).
-> verify: `git log --follow studies/data-scaling-law/analysis/rl_baseline.py`
   in dispatcher still shows history up to the removal commit.

### Step 6 — Update Beaker / CO workflows

Anywhere the CO capsule or Beaker templates hardcode `studies/...` paths,
either:
- update the path to point at `../aind-disrnn-studies/...` (side-by-side
  layout also inside the container), or
- clone the studies repo inside the container entrypoint.

Verify by launching one small variant end-to-end after the split.

## Verification checklist (end-to-end)

Run in `aind-disrnn-studies/studies/data-scaling-law/` after the split:

```bash
source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu
make all      # regenerates r1, r3, r4, r5, r7, r8 outputs
```

- [ ] `make all` exits 0.
- [ ] Regenerated JSONs are byte-identical to pre-split (aside from
      `_meta.produced_at_pt`/`dispatcher_git_sha`).
- [ ] `reports/r7-nxd-joint-scaling-grid.md` and `reports/r8-gru-vs-rl-baseline.md`
      re-render identically between the `BEGIN/END` markers.
- [ ] `git log --follow analysis/rl_baseline.py` shows commits going back to `1e30716`.
- [ ] Dispatcher `git status` clean after the removal PR merges.
- [ ] One trial launch from the studies repo (`launch_beaker_resumable.py
      --help` at minimum) succeeds pointing at sibling dispatcher.

## Studies-repo `AGENTS.md` template

```markdown
# AGENTS.md — aind-disrnn-studies

Behavioural rules for this repo. Framework-wide rules (HPC safety,
Conventional Commits, PR merge policy, Beaker scheduling, verify-with-data,
posthoc-analysis, human-facing logs) live in the sibling repo
`aind-disrnn-dispatcher` at `AGENTS.md` and are inherited by reference —
do not duplicate here.

## Studies-repo-specific rules

- Every study is a subfolder of this repo root, laid out per
  [`aind-disrnn-dispatcher/docs/study-organization.md`](../aind-disrnn-dispatcher/docs/study-organization.md).
- Post-hoc analysis and reporting: per
  [`aind-disrnn-dispatcher/docs/posthoc-analysis.md`](../aind-disrnn-dispatcher/docs/posthoc-analysis.md).
- Launch commands assume `../aind-disrnn-dispatcher/` exists as a sibling
  clone. See top-level `README.md` for the layout.
```

## Non-goals (explicitly out of scope)

- **Do not** promote the `code/` launchers to a pip-installable package
  in this migration. That's a separate refactor; do it after the split
  proves stable.
- **Do not** split individual studies into their own repos. One studies
  monorepo suffices until we have >5 studies or a clear ownership boundary.
- **Do not** move `docs/` into the studies repo. Framework conventions
  belong with the framework; studies link back.
- **Do not** rewrite dispatcher history (only `studies/` is filter-repo'd,
  and only in the extract clone).

## Open questions (resolve before executing, or in-line)

1. **Path prefix in the new repo.** Flat (`data-scaling-law/` at root) vs.
   prefixed (`studies/data-scaling-law/`)? This plan assumes prefixed for
   minimum diff. Flat is cleaner long-term.
2. **`.codeocean/` capsule scope.** Does the CO capsule need the studies
   folder present to function, or only the launchers? If the former, either
   ship a submodule/subclone step in the capsule entrypoint or accept that
   post-split the capsule no longer builds studies.
3. **`code/config/`, `code/beaker/` templates.** Some templates
   (`sweep_scaling.yaml`, `sweep_gru_scaling.yaml`) are named after
   data-scaling-law but live in dispatcher. Do they move with the studies
   extract, or stay as reusable dispatcher templates? Recommend: stay in
   dispatcher for now; extract to study-specific `variants/*/sweep.yaml`
   only if a second study needs a different one.
4. **`aind-disrnn-wrapper` version pin.** The current
   `studies/data-scaling-law/environment.lock` pins the wrapper at commit
   `870e6dd`. After split, that pin migrates with the study. Should the
   studies repo also pin dispatcher itself (e.g. via a
   `dispatcher_commit_sha` file in the study root)? Recommend: yes,
   one-line file `.dispatcher_pin` alongside `environment.lock`.
5. **CI.** Dispatcher has no `.github/workflows/` today. If CI is added
   before the split, does it run study Makefiles? If yes, split it too.
6. **Existing branches on dispatcher.** After `studies/` is removed from
   `main`, any long-lived feature branch that touched `studies/` becomes
   painful to merge. Enumerate open branches first
   (`gh pr list --state open --json headRefName,files`), rebase or close
   before Step 5.

## Related

- [[study-organization]] — the intra-study layout that already anticipates
  this split ("one folder per study", "self-contained variants").
- [[posthoc-analysis]] — analysis conventions that already live *inside*
  each study folder and travel with it.
- [[beaker-playbook]] — launcher-side conventions that stay in dispatcher.
- [[AGENTS]] §5 (HPC safety), §6 (Conventional Commits), §9 (PR merge
  policy, no squash), §10 (Beaker), §11 (verify-with-data), §12 (posthoc).
