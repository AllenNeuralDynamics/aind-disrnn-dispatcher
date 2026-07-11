# launch_record — validation-2way-vs-3way

- **W&B group:** `validation-2way-vs-3way@20260703-092118` (2 runs, D≈10/seed-0,
  `data.ignore_policy ∈ {exclude, include}`). **The runs have since been pruned**
  from W&B in a stale-run cleanup — this was a throwaway smoke, superseded by the
  full `nxd-3way` grid; the launch spec + this record are the surviving provenance.
- **W&B project:** https://wandb.ai/AIND-disRNN/mice_ignore_scaling
- **Beaker exp:** `01KWMCHPQ803BCBR0AKP4VZWDW` (2-task resumable, g6e L40S, 1-GPU
  bundle). Dispatcher commit `ad9fb6ef20fe8a1112dbcd98d39835288a53afb2`.
- **Submitted:** 2026-07-03 09:21 PT
- **Settled:** 2026-07-03 PT (short smoke, `n_steps=20000`)
- **Status:** ✅ success — both cells reached exit 0; the `include` cell logged a
  3-class output and completed the held-out fine-tune, confirming the 3-way
  (`output_size=3`, L/R/ignore) pipeline trains end-to-end on Beaker.
- **Key numbers:** first-look only (short training, not comparable to the grid).
  The `exclude` cell tracked toward the familiar ~0.72 L/R held-out likelihood; the
  `include` cell's raw 3-way likelihood (chance 1/3) is **not** comparable by
  subtraction — the like-for-like comparison (conditional L/R on engaged trials +
  ignore-class separately) was deferred to the analysis pass on the full grid.
- **Feeds reports:** none directly — this variant validated the pipeline; all
  reported numbers come from `nxd-3way`.
- **Notes.** Purpose was pipeline validation, not a result. Its job done, its runs
  were pruned; keep this record + `sweep.yaml` for the audit trail of "when did the
  3-way path first run green."
