# Variant nxd-h512

**Status.** ✅ Shipped, feeding r7. D=614 excluded — logged as an unattainable
cell (host-RAM OOM, unresolved root cause), not a data gap to backfill later.

**What differs from the siblings.** Adds a new capacity row `hidden_size=512` to
the N×D joint-scaling grid (r7). Every other knob is identical to `nxd-grid` /
`v2-sc-active` (scalar session conditioning, λ-forward full SC @50k,
`n_steps=150k`, `lr=1e-5`, gated early-stop @70k, length bucketing).

**Why.** r7 shows D saturates by ~100 mice and the N×D interaction is weak up to
H=256. This tests whether that story holds one capacity octave higher (H=256→512).

## Timeline

**Launch 1 (a100) — FAILED, all 12 crashed at step 0.** sweep `5iy5qnb2`, group
`nxd-h512@20260718-192434`, array `23241259`. GPU OOM (`RESOURCE_EXHAUSTED`) — the
`aind` a100s are full 40 GB PCIE cards and H=512 at `batch_size=2048` exceeds that
at peak. Nothing to salvage (W&B confirms `_step=0` on all 12).

**Launch 2 (h200) — 9/12 landed.** sweep `ajsw1a8h`, group
`nxd-h512@20260720-195322`, array `23263174`. D=10/30/100 × 3 seeds (9 cells)
finished cleanly on h200 (full 141 GB fits H=512). The 3 **D=614** seeds
(`subject_ratio=1.0`, full 614-mouse dataset) failed with a **host-RAM OOM**
(`slurmstepd oom_kill`, MaxRSS 81–95 GB against a 128G cgroup) — not a GPU issue.

**D=614 recovery attempts — all failed, root cause never resolved.**
- Resubmit at `--mem=256G` (sweep `aw04872p`, array `23302804`): OOM'd again,
  MaxRSS 230–257 GB and *still climbing* when killed — ruled out "just needs a
  bit more memory."
- RAM-profiling investigation (throwaway CPU probes on `aind_debug`, never the
  login node): decomposed the RSS trace into two candidate mechanisms. The
  length-bucketing JAX-compile-cache hypothesis was **falsified** by a control
  probe with bucketing disabled — it OOM'd identically. A JAX async-dispatch
  backpressure hypothesis (added `jax.block_until_ready` per step) was also
  **falsified** — OOM'd at native speed too, confirmed the edit was genuinely
  live. Three hypotheses tried; the leak was never localized to a specific line.
- **Beaker reproduction (decisive):** relaunched the 3 D=614 cells on Beaker
  h200 (experiment `01KYWJ3JYVE7BJVEVVM5CCT6Q5`, matching the known-good
  H=256/D=614 spec exactly except memory). OOM'd again, at a similar wall-clock
  to the HPC probes. This ruled out both the SLURM/cgroup backend and the CPU
  vs GPU backend as the cause — **the leak is backend-independent**, tied to
  something in the D=614 training path itself (18,124 sessions vs ~3,000 at
  D=100 is the only structural difference; the mechanism was never isolated
  beyond that).

**Decision (Han, 2026-08):** stop debugging. Log the D=614 OOM as an
unattainable cell and ship the H=512 row as **D ∈ {10, 30, 100} only**, exactly
like D=300 is already excluded from r7 for a different reason (grid
rectangularity). Unlike D=300 (real data, just not plotted in the joint grid),
D=614/H=512 genuinely does not exist — no successful run at that cell on any
backend tried (HPC SLURM at 128G/256G, Beaker at 256G).

**Held-out metric backfill (separate bug, fixed).** The 9 surviving cells
initially carried no `heldout/final/eval_likelihood` at all — the end-of-training
auto held-out fine-tune silently failed with `[Errno 13] Permission denied:
'/results'` (that path exists in the Beaker/Code Ocean container, not on HPC).
This was a **second occurrence** of a bug already patched once for study-02's
sweeps only; fixed properly this time in `launch_hpc.py` (PR #68, merged to
`main`, auto-injects `auto_heldout_finetune.output_root` on every HPC sweep).
The 9 already-trained cells were backfilled from their saved checkpoints
(SLURM array `24991820`, ~1 h/cell, no retraining) rather than relaunched.

## Corrected finding (retraction)

An earlier read of this row's results used the wrong metric (a top-level
`likelihood` field — an in-training diagnostic, not the held-out adaptation
metric) before the backfill above existed, and concluded H=512 "overfits
catastrophically" at low D. **That claim is retracted.** With the correct
`heldout/final/eval_likelihood`, H=512 shows no overfitting and tracks H=256
closely:

| D | H=512 | H=256 |
|---|---|---|
| 10 | 0.7213 | 0.7214 |
| 30 | 0.7253 | 0.7251 |
| 100 | 0.7278 | 0.7273 |

A normal, modest continuation of the N-scaling trend — not a capacity-vs-data
crisis. See `fig_nxd_scaling.png` / `reports/r7-nxd-joint-scaling-grid.md`.

## Final state

**W&B group feeding r7:** `nxd-h512@20260720-195322` (9 cells, D∈{10,30,100}),
project `AIND-disRNN/mice_data_scaling`. The a100 group
(`nxd-h512@20260718-192434`, all-failed) and the D=614 groups
(`nxd-h512@20260723-125804`, `01KYWJ3JYVE7BJVEVVM5CCT6Q5`, both all-failed)
have nothing to collect and are not in `NXD_GROUPS`.

**Analysis wiring — done.** `analysis/nxd_scaling.py`: `TARGET_HS` includes 512;
`NXD_GROUPS` includes the group above. `analysis/update_final_report_nxd.py` was
patched to render the missing D=614 cell as "n/a" instead of crashing on the
ragged grid (a `None - float` TypeError the first regen attempt hit). r7 and its
figure were regenerated on an HPC compute node (`aind_debug`, never the login
node — see AGENTS.md §5) and committed.
