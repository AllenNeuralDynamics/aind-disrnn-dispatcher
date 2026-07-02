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
