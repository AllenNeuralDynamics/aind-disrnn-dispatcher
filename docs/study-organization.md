# Study & experiment organization

Conventions for organizing studies, variants, and run provenance. The one-line rule lives in
`AGENTS.md` §8; this file holds the full scheme.

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

## Provenance / tracking (one launch == one "pseudo-sweep")

Every launch is uniquely and *readably* identifiable, with platform-native ids saved
alongside for cross-ref:

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
