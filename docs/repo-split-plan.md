---
aliases:
  - repo split plan
  - split studies
  - aind-disrnn-studies
tags:
  - planning
  - migration
  - one-shot
status: approved
---

# Repo-split plan: extract `studies/` into `aind-disrnn-studies`

> **Status:** approved 2026-07-16, not yet executed. Delegated to a separate
> agent for execution. Originally written 2026-06-30 after the
> `posthoc-analysis` standard-structure migration (ending at commit `122abfd`,
> later merged to `main`); revised 2026-07-16 when the split was approved.

## TL;DR

Split `aind-disrnn-dispatcher` at the framework/application seam:

- **`aind-disrnn-dispatcher`** (this repo, stays) — launchers (`code/`),
  framework docs (`docs/`), Docker env, CO capsule metadata, root `AGENTS.md`.
- **`aind-disrnn-studies`** (new sibling repo) — everything currently under
  `studies/`, keeping the `studies/` path prefix (see Decision below), one
  study per subfolder, each self-contained per `docs/study-organization.md`.

All five studies (`01`–`05`) move in one pass. Preserve per-file history via
`git filter-repo`.

## Decision (2026-07-16): split now

The 2026-06-30 discussion below leaned "not yet," with the explicit revisit
trigger *"when there is a second study."* Re-measured 2026-07-16, every
load-bearing fact has flipped:

- **Five studies exist** (`01-gru-scaling-law` … `05-disrnn-scaling-law`),
  each with its own `Makefile`, `analysis/`, and reports.
- **Studies dominate the repo.** Last 3 months: 255 non-merge commits touched
  `studies/` vs ~175 file-touches under `code/`. Tracked `studies/` content is
  ~43 MB vs ~3 MB for everything the runtime needs. Framework changes are
  drowned out, and every study commit advances `main`, so pinned
  `DISPATCHER_REF` SHAs churn for reasons meaningless to the runtime. Beaker
  jobs re-clone the dispatcher at startup (`entrypoint.sh`), so the split also
  keeps that clone permanently light.
- **Coupling is thin.** Of the 255 study commits, only 11 (~4%) also touched
  `code/` or `environment/` — cross-repo PRs will be rare. Verified again:
  zero Python imports from `studies/**` onto `code/**`; the only references
  are shell-path launch commands in READMEs/notes and comment lines in sweep
  YAMLs. (Study 04's analysis scripts `sys.path`-insert the *wrapper* sibling
  clone — cross-repo side-by-side consumption is already the working
  convention.)
- **Packaging `code/` is NOT a prerequisite** (reverses the 2026-06-30
  sequencing note below, restoring the original Non-goals bullet). The real
  interface is a CLI that takes the `sweep.yaml` path as an argument;
  `launch_hpc.py` and `launch_beaker_resumable.py` only parse the
  `studies/<study>/variants/<variant>` components out of that path to derive
  the W&B group. A sibling clone
  (`python ../aind-disrnn-dispatcher/code/launch_....py studies/...`) works
  today with no code change. Packaging remains a good follow-up, not a
  blocker.
- **Path prefix (was Open Question Q1): keep `studies/` — resolved, and not
  just for lower churn.** The launchers' study/variant derivation searches for
  the literal `studies` path component (`launch_hpc.py:218`,
  `launch_beaker_resumable.py:135`); a flat layout silently breaks W&B group
  naming unless both launchers are patched. Prefixed it is; flatten later only
  together with a launcher change.

Execution notes: the sandbox cannot create `.git` directories, so
`gh repo create` and the initial clones on the Mac and HPC checkouts are the
user's steps; everything else is delegable. Prereq 1 below re-verified
2026-07-16: no open PRs, `git log origin/main..HEAD -- studies/` empty.

## Should we do this now? (discussion 2026-06-30 — the "not yet" lean is SUPERSEDED by the 2026-07-16 decision above; the analysis-layer findings and contract hardening below still stand)

**Lean at the time: not yet.** The split is a defensible long-term direction but
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
├── studies/                     # prefix KEPT — see resolved Q1
│   ├── 01-gru-scaling-law/
│   │   ├── analysis/
│   │   ├── variants/
│   │   ├── Makefile
│   │   ├── environment.lock
│   │   ├── CHANGELOG.md
│   │   └── README.md
│   ├── 02-gru-scaling-law-ignore/
│   ├── 03-disrnn-beta-scan/
│   ├── 04-gru-vs-disrnn-embedding-recovery/
│   ├── 05-disrnn-scaling-law/
│   └── util/
└── .gitignore                   # ignore per-study analysis/_cache_*.json, etc.
```

Path prefix trade-off (resolved 2026-07-16, see Decision above and Q1):
- **Flat** (`01-gru-scaling-law/` at root): tighter for a repo whose sole
  purpose is studies — but **breaks the launchers' W&B group derivation**,
  which searches for the literal `studies` path component.
- **Prefixed** (`studies/01-gru-scaling-law/`): zero-diff paths for docs,
  Makefiles, launchers, and tools that hardcode the `studies/` prefix; easier
  `filter-repo`. **Chosen.** Flatten later only together with a launcher
  change.

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

Files to update in the studies repo after extraction: **regenerate the list at
execution time** —

```bash
rg -l 'code/launch|code/config|code/hpc|code/beaker' studies/
```

As of 2026-07-16 this hits ~20 files across studies 01–05: per-study
`README.md`s, `variants/*/notes.md`, `variants/*/sweep.yaml` and
`variants/*/experiment.yaml` (comment lines only),
`variants/*/launch_record/*.yaml` (comment lines only),
`01-gru-scaling-law/launch_heldout_rerun.py` (docstring),
`01-gru-scaling-law/analysis/rl_baseline_verdict.md`, and
`01-gru-scaling-law/analysis/rl_baseline.py` (an error-message string). Nearly
all are docs/comments; only the last two are code strings.

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
history preserved (`git log --follow studies/01-gru-scaling-law/analysis/rl_baseline.py`
should show every commit that touched it, across the
`data-scaling-law` → `01-gru-scaling-law` rename — verified working on the
source repo 2026-07-16).

-> verify: `git log --oneline | wc -l` in `extract/` is > 250 (255 non-merge
   commits touched `studies/` in the last 3 months alone).
-> verify: `git log --follow --oneline studies/01-gru-scaling-law/analysis/rl_baseline.py`
   shows at least commits `1e30716`, `342f5ae`, `122abfd`, etc.

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
-> verify: `git log --follow studies/01-gru-scaling-law/analysis/rl_baseline.py`
   in dispatcher still shows history up to the removal commit.

### Step 6 — Update Beaker / CO workflows

Anywhere the CO capsule or Beaker templates hardcode `studies/...` paths,
either:
- update the path to point at `../aind-disrnn-studies/...` (side-by-side
  layout also inside the container), or
- clone the studies repo inside the container entrypoint.

Verify by launching one small variant end-to-end after the split.

## Verification checklist (end-to-end)

Run in `aind-disrnn-studies/` after the split — every study has a `Makefile`;
study 01 is the deepest regeneration test, but run all five:

```bash
source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu
for s in studies/0*/; do (cd "$s" && make all); done
```

- [ ] `make all` exits 0 in all five studies.
- [ ] Regenerated JSONs are byte-identical to pre-split (aside from
      `_meta.produced_at_pt`/`dispatcher_git_sha`).
- [ ] Study 01's reports re-render identically between the `BEGIN/END`
      markers.
- [ ] `git log --follow studies/01-gru-scaling-law/analysis/rl_baseline.py`
      shows commits going back to `1e30716`.
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

## Open questions (Q1, Q2, Q4 resolved 2026-07-16; Q3, Q6 resolve at execution)

1. **Path prefix in the new repo.** ~~Flat vs. prefixed?~~ **RESOLVED:
   prefixed** (`studies/01-gru-scaling-law/`). Not just lower churn — the
   launchers derive the W&B group by finding the literal `studies` path
   component (`launch_hpc.py:218`, `launch_beaker_resumable.py:135`); flat
   silently breaks group naming. See Decision section.
2. **`.codeocean/` capsule scope.** **RESOLVED: launchers only.** Verified
   via `.codeocean/app-panel.json`: the capsule's Reproducible Run executes
   `code/launch_beaker.py` against `code/beaker/sweep_mvp.yaml` and writes a
   launch record to `/results` — it never touches `studies/`. Nothing to
   change in the capsule.
3. **`code/config/`, `code/beaker/` templates.** Some templates
   (`sweep_scaling.yaml`, `sweep_gru_scaling.yaml`) are named after
   study 01 but live in dispatcher. Do they move with the studies
   extract, or stay as reusable dispatcher templates? Recommend: stay in
   dispatcher for now; extract to study-specific `variants/*/sweep.yaml`
   only if a study needs a diverging copy.
4. **`aind-disrnn-wrapper` version pin.** Each study's `environment.lock`
   pins the wrapper; after the split those pins migrate with the studies.
   **RESOLVED: yes**, the studies repo also pins dispatcher — one-line file
   `.dispatcher_pin` alongside each `environment.lock`, stamped at launch
   time like `_meta.dispatcher_git_sha` already is in analysis JSONs.
5. **CI.** Dispatcher has no `.github/workflows/` today. If CI is added
   before the split, does it run study Makefiles? If yes, split it too.
6. **Existing branches on dispatcher.** After `studies/` is removed from
   `main`, any long-lived feature branch that touched `studies/` becomes
   painful to merge. Enumerate open branches first
   (`gh pr list --state open --json headRefName,files`), rebase or close
   before Step 5. (Checked 2026-07-16: zero open PRs — re-check at
   execution time.)

## Code Ocean: launch surface and archive, not a home (decided 2026-07-16)

Considered and rejected: migrating `studies/` + the Makefile mechanism into
the Code Ocean ecosystem. The current division of labor is correct and the
split does not change it — CO = launch control plane (the capsule's app panel
runs `launch_beaker.py`), Beaker/HPC = compute, W&B = experiment record,
git = code and living documents. Three reasons:

- **Wrong data plane.** Every analysis producer opens `wandb.Api()` and pulls
  by group name; there is no local-data input. CO's core value (immutable
  mounted data assets, lineage) buys nothing when the input is a cloud API —
  a capsule "Reproducible Run" would still depend on mutable external W&B
  state. The provenance actually relied on (`_meta` git SHAs,
  `environment.lock`, W&B group names) already travels with git.
- **Wrong iteration model.** The reports-are-code loop — edit producer,
  `make rN` in seconds against `.gitignore`'d W&B caches, diff the
  regenerated `BEGIN/END` region, review in a PR — is incremental and
  git-native. A capsule has one entrypoint, no incremental targets, no cache
  persistence between runs, and immutable per-run results. Study history is
  full of `fix(study-04): correct overstated claim` commits: reports are
  living documents corrected under review, git's home turf.
- **A fourth home that retires none of the other three,** for a
  2-person-plus-AI team with no external consumers of intermediate state.

**Where CO does earn its place — a hermetic analysis-regeneration capsule**
(analysis only; it never triggers Beaker/HPC — that stays the existing launch
capsule's job). Because studies 03–05 already follow the "freeze the numbers"
convention (committed curated grid CSVs; `make all` runs offline), a capsule
that clones the studies repo at a pinned SHA and runs every study's `make
all` needs no W&B secret, no network, and no data asset. Design, priorities
(CO explicitly last), and the remaining normalization work (study 01,
study 05 r4): [[report-publication-and-reproducibility]]. Out of scope for
this split.

## Committed artifacts policy: keep committing figures & CSVs (decided 2026-07-16)

Measured 2026-07-16: `studies/` tracks 45 PNGs (21 MB) and 46 CSVs (5.2 MB);
largest single files ~1–1.5 MB (study 04 JSONs/CSVs). Decision: **keep
committing them, in the new studies repo.** Rationale:

- Reports embed the figures; without committed PNGs the `r*.md` reports are
  unreadable on GitHub, defeating reports-are-code.
- Regenerated-artifact diffs are a real review signal (history shows them
  catching dropped CSV rows and overstated claims).
- "Regenerable from W&B" decays: if runs are deleted or age out, the
  committed artifacts are the only durable record of a study's answer.
- The dispatcher-weight concern is solved by the split itself — study
  artifacts no longer ride along in the runtime clone that Beaker jobs pull
  at startup.

Discipline in the studies repo: W&B pull caches stay `.gitignore`'d (never
commit them); keep single artifacts under a few MB (downsample figures,
aggregate CSVs — raw pulls belong in the cache layer); revisit git-LFS only
if the repo's pack size approaches a few hundred MB, which at the current
~27 MB of history-blob weight is years away.

## Related

- [[report-publication-and-reproducibility]] — the post-split publication
  layers (freeze-the-numbers completion, CI regen check, docs site, CO
  analysis capsule) that build on this split.
- [[study-organization]] — the intra-study layout that already anticipates
  this split ("one folder per study", "self-contained variants").
- [[posthoc-analysis]] — analysis conventions that already live *inside*
  each study folder and travel with it.
- [[beaker-playbook]] — launcher-side conventions that stay in dispatcher.
- [[AGENTS]] §5 (HPC safety), §6 (Conventional Commits), §9 (PR merge
  policy, no squash), §10 (Beaker), §11 (verify-with-data), §12 (posthoc).
