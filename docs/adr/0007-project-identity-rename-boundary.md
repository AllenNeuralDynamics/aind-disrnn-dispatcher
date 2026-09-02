# The project renames to dynamic-foraging-bfm; the disRNN architecture and the run record do not

The project outgrew its name. Studies 01 and 07 are GRU, 08 is HB-vs-GRU, and the tracking
issue is already titled "Behavioral Foundation Model for Dynamic Foraging" — but the repos,
the W&B entity, the conda envs and the env-var contract still say `disrnn`. The **project
identity** renames to `dynamic-foraging-bfm`:

| Thing | From | To |
|---|---|---|
| Repos | `aind-disrnn-{dispatcher,wrapper}` | `aind-dynamic-foraging-bfm-{dispatcher,wrapper}` |
| W&B entity | `AIND-disRNN` | `AIND-dynamic-foraging-bfm` |
| Conda envs | `disrnn-{cpu,gpu}` | `dynamic-foraging-bfm-{cpu,gpu}` |
| Env-var prefix | `DISRNN_*` | `BFM_*` |
| Python distribution | `aind-disrnn-wrapper` | `aind-dynamic-foraging-bfm-wrapper` |

`aind_disrnn_utils` was a third owned repo when this rename was scoped. It has since been
retired rather than renamed: its code is vendored into the wrapper and the wrapper's
`pyproject.toml` no longer depends on it. The repo stays for history; it is not renamed.

### Why `dynamic-foraging-bfm` and not `behavior-fm`

`behavior-fm` was the working name and is what the first cut of this ADR carried. It is
wrong for AIND specifically: the institute runs several behaviour paradigms (VR foraging
among them), so "the AIND behaviour foundation model" claims a scope this project does not
have. Everything here is rodent dynamic foraging, and the generalisation being tested is
across *subjects within that task* — held-out subject, zero-shot, few-shot — not across task
families. `aind-dynamic-foraging-bfm-*` also joins the naming family this repo already
depends on five times over (`aind-dynamic-foraging-models`, `-data-utils`,
`-basic-analysis`, `-multisession-analysis`, `-database`).

The `BFM_` env-var prefix is unaffected by the choice — it reads as "behaviour foundation
model" either way — so nothing in the expand-contract migration below depends on it. If the
model ever trains on other behaviours, this name becomes the misnomer; that trade was made
knowingly, on the evidence that every study to date (01–08) is dynamic foraging.

## The boundary

`disrnn` names two unrelated things here, and only one of them is the project.

**Model architecture — frozen, never renamed.** `DisrnnTrainer`, `MultisubjectDisRnn`,
`models/disrnn_network.py`, `disrnn_config`, `create_disrnn_dataset`, and the
`disentangled_rnns` / `disrnn.information_bottleneck` imports name the disentangled-RNN
architecture, which sits *alongside* GRU and HB in this codebase. Renaming them would make
the code less accurate, not more — there would be no way left to say which of the three
model families a symbol belongs to.

**Historical provenance — frozen, never renamed.** Everything under
`studies/*/launch_record/` and `studies/*/analysis/`, the report files, all existing W&B
project and entity names, already-built Beaker image tags, and study folder names 03–06
record what actually ran. Rewriting them falsifies history and breaks every link from a
finished report to the run that produced it.

**Out of scope.** `aind-disentangled-rnns` is a fork of `google-deepmind/disentangled_rnns`
with live upstream merges. Renaming a fork you still pull from is friction for zero gain.

## Consequences

The rename is bounded because the frozen set is most of it. Verified at `6b58d8b`: of 4,843
`disrnn` occurrences across 474 dispatcher files, 4,385 are under `studies/` and 3,821 of
those sit in the frozen `launch_record/` and `analysis/` paths. The live surface is 243
occurrences in `code/`, 125 in `docs/`, and 69 of the 1,054 `DISRNN_*` env-var occurrences
(the other 985 are frozen study records). Counts drift; the ratio is the durable point.

The cost of not writing this down is specific: the next agent asked to "finish the rename"
runs a global find-replace, renames `DisrnnTrainer`, and rewrites `studies/*/launch_record/`
— destroying the provenance linking every report to its runs. A CI guardrail enforces this
boundary mechanically; this ADR is what it encodes.

The env-var prefix migrates expand–contract (wrapper reads `BFM_*` alongside legacy
`DISRNN_*` → dispatcher emits `BFM_*` → wrapper drops the legacy read) rather than as a
flag-day switch, because the dispatcher and the wrapper deploy independently. GitHub keeps
redirects after a repo rename, so pinned SHAs and existing clones keep resolving throughout.

Tracked by #74.
