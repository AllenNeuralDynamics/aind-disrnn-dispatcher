# nxd-3way

**What differs.** Exact mirror of `data-scaling-law/variants/nxd-grid`, with the single change
`data.ignore_policy=include` (⇒ `output_size=3`, L/R/ignore). Full 4×4 grid in one variant:
`hidden_size ∈ {16,64,128,256}` × `subject_ratio ∈ {0.016,0.049,0.163,1.0}` (D≈10/30/100/614) ×
`seed ∈ {0,1,2}` = **48 tasks**. All training knobs identical to `nxd-grid`/`v2-sc-active`.

**Why it exists.** The r7 grid showed D saturates by ~100 mice on 2-way L/R likelihood at every N.
This runs the same grid on the 3-way (engagement-aware) target to test whether a headroom-ier metric
shows a D-slope that persists past 100 — the core hypothesis of this study (roadmap #23).

**Compute.** Routed to `octo-hub-onprem-h200` (H256 needs 141GB; a 48GB L40S OOMs on the wide model).
If contended, render `--no-submit` and split H16/H64 tasks to g6e L40S per the beaker-launch skill.

**Launch (after validation passes):**
```bash
conda activate disrnn-cpu
WS=ai1/aind-dynamic-foraging-foundation-model
python code/launch_beaker_resumable.py \
  --sweep studies/ignore-trials-scaling/variants/nxd-3way/sweep.yaml \
  --experiment studies/ignore-trials-scaling/variants/nxd-3way/experiment.yaml \
  --workspace "$WS" \
  --label nxd-3way \
  --note "N×D grid with ignore_policy=include (3-way output); test whether engagement target keeps scaling in D past ~100 mice (r7 mirror)"
```

**Status.** 🚧 pending validation-2way-vs-3way. W&B group `nxd-3way@<launch_id>`, project `mice_ignore_scaling`.

**Result.** _TBD._
