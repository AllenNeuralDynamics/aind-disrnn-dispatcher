# Backfill history — why 12 cells were re-scored

Audit trail for a correctness fix applied to the `nxd-3way` grid. Human-readable;
the machine-readable launch record is `rescore_launched.json`.

## The bug: restore-based backfill over-trained the model

When the ignore-class metrics (`engage_ignore_*`) were added to the wrapper
*after* part of the grid had already trained, the missing cells' ignore metrics
were "backfilled" using the `DISRNN_RESTORE_FROM_RUN_ID` restore path. That path
**resumes base training** from the saved checkpoint — and early-stop patience
resets on resume, so each backfilled run trained ~30k extra base steps past its
original early-stop. The result was a **different, over-trained model**: its
held-out numbers (both LR-engaged and the new ignore metrics) do not match the
native run.

Symptom that surfaced it: D=10/H=256/seed0 LR-engaged dropped from the native
**0.71841** to **0.64816** in the restore run — a catastrophic outlier. Most
other cells drifted only ±0.001–0.008, small enough to nearly pass unnoticed.

## Affected cells: exactly 12

A cell is affected iff it had BOTH a native run (lower step) AND an
ignore-bearing run at a restore-signature step (native_step + 30000, except
D10/H256/s0 which is +20000). The 12:

- D=10/H64 s0,s1,s2 · D=30/H64 s0,s1,s2
- D=10/H16 s0,s1 · D=30/H16 s0,s1,s2
- D=10/H256 s0

The other 36 cells got their ignore metrics natively at end-of-training (the
Beaker hedge for D=100/D=614, plus D30/H128 and D30/H256/s0 fresh-trained) and
were never touched.

## The fix: exact held-out re-score (no base training)

`resume_heldout_beaker.py` (wrapper commit `0a9141b`) re-runs the held-out
finetune-and-eval **only** — it downloads the native run's training-output
artifact, reconstructs `inputs.yaml` from the W&B run config, resolves the
`best_eval` checkpoint, runs the 500-step embedding-only held-out finetune, and
writes the correct metrics back into the native run (`resume="must"`). No base
training, so the model is bit-identical to the native end-of-training state.

Three commits were needed to make the Beaker port faithful:

1. `c757463` — the port itself (HPC `resume_heldout.py` → Beaker).
2. `82e6539` — fetch run config via direct GraphQL (the container's `wandb.Api().run()`
   raises `Object of type Api is not JSON serializable` on the sweep auto-load).
3. `0a9141b` — **the decisive fix.** The artifact's `checkpoints/index.json` records
   each `params_path` as the original absolute HPC path
   (`/allen/.../han.hou/outputs/disrnn/.../files/outputs/checkpoints/step_N/params.json`),
   which contains `/outputs/` **twice**. `resolve_model_run`'s `_resolve_artifact_path`
   splits on the *first* `/outputs/`, yielding a nonexistent path, so `best_eval`
   silently fell back to `final` (step_90000) → wrong checkpoint → LR 0.71284 instead
   of 0.71987. The driver now rewrites those paths to relative form before resolving.

## Validation

`j2uh7x44` (D10/H64/s0) re-score reproduced the HPC-proto target exactly:
LR-engaged **0.7198663** vs **0.7198668** (|Δ| = 4.65e-07); all ignore metrics
matched. Only after this passed were the other 11 cells re-scored.

## Downstream

`analysis/scaling.py`'s dedup was corrected in the same effort: among
ignore-bearing runs for a cell it now prefers the **lowest** `_step` (the native
run), discarding the over-trained restore run. See the comment at `fetch_3way`.

> **Care note for future re-scores.** `resume="must"` OVERWRITES the target run's
> `heldout/final/*` summary. A *wrong* re-score clobbers the correct native value
> (this happened once to `j2uh7x44`, then was repaired). Always validate that a
> re-score reproduces the target on one cell before running it across the grid.
