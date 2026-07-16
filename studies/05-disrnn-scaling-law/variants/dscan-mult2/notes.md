# Variant: dscan-mult2 — the disRNN data-scaling curve (primary)

**What differs.** The study's main grid: `data.subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}`
(D ≈ 10/30/100/300/614) × `seed ∈ {0,1,2}` = **15 tasks**, at study 03's recommended operating
point (mult=2, β=1e-3, lr=1e-3). Everything else is held fixed at study 03's protocol, so D is
the only axis that moves.

**Why this operating point** (study 03's verdict, established at D=100):
- **mult=2** compresses the interaction bottleneck ~2.7× at no held-out cost while leaving ~1 open
  channel to interpret. mult=5/10 collapse it to *zero* open channels — nothing left to read.
- **β=1e-3** (moderate) over 3e-4 (weak): held-out transfer is flat between them, but 1e-3 gives a
  tighter interaction bottleneck (openness 0.81 vs 1.16) at the same transfer.
- **lr=1e-3**, not 5e-3: both mult=10 NaN divergences in study 03 were at lr=5e-3.

**Comparability to the GRU curve (study 01).** Same D ladder, same *fixed* held-out cohort
(`heldout_every_n=5` — mice never trained on, identical at every D and seed), same snapshot
(20260603), same 2-way output (`ignore_policy=exclude`), same batch size, and the same y-axis key
`heldout/eval_likelihood` from the final `auto_heldout_finetune`. So the disRNN curve overlays the
GRU curve on identical axes and the two power-law fits are directly comparable.

**Expected.** Two read-outs, one of which is new:
1. **Transfer vs D** — the GRU curve saturates by ~100 mice (+0.005 total from D=10→614). If the
   disRNN saturates the same way, that supports "the metric, not the architecture, is the
   bottleneck" (study 01's verdict). If it keeps climbing past 100, the disRNN's inductive bias is
   using the extra mice and the saturation was GRU-specific — the more interesting outcome.
2. **Bottleneck openness vs D** — study 03's motivating worry is that the interaction bottleneck
   "does not sparsify when many mice are trained together", but that was only ever checked *at*
   D=100, never *across* D. This grid measures openness Σ(1−σ) as a function of D at fixed β and
   multiplier: does interpretability actually degrade as the cohort grows?

**Caveat to carry (SC window).** `n_steps=60000` with full session conditioning at 50k leaves 10k
full-SC steps — study 03's schedule, kept deliberately so the D-comparison here is internally
clean and cell-comparable to 03. But the GRU curve had 100k post-SC steps, and study 01 found the
SC gain *grows with D*. So a flat disRNN D-curve is only weak evidence against SC-at-scale: the
window may be too short for the large-D SC gain to materialize. If the curve is flat, the
follow-up is an extended-horizon arm (`restore_from_run_id` → 150k) at D∈{100,614}, not a new
study. Do not read the flat case as "disRNN can't use more mice" without that check.

**Metric caveat (inherited from study 03).** Report bottleneck openness as
**`total_openness` = Σ(1−σ)**, *not* `n_eff_open_frac` — the participation-ratio metric is
scale-invariant and reports a spuriously high value even for a fully shut bottleneck (it
mis-ranked 19/43 runs in study 03). See `03-disrnn-beta-scan/analysis/provenance/metric_caveat.md`.

**Cost.** 15 tasks × ~18 h (measured at D=100 on study 03's grid; per-step cost is D-independent
at fixed `batch_size=2048`) ≈ **270 GPU-h**, low-preemptible on g6e L40S.

**Status.** ⏳ running 15/15 — launched 2026-07-13 00:35 PT, group `dscan-mult2@20260713-003428`,
Beaker [`01KXD6CDKKN2CARG16AW4XQRJN`](https://beaker.org/ex/01KXD6CDKKN2CARG16AW4XQRJN). The
(D=614, seed=2) cell died pre-start on the bad node `aidc-h200-prd2` and was re-submitted verbatim
(same run id + group) off that cluster as
[`01KXD6PA22ZZW2MJ2CH0JSKSWT`](https://beaker.org/ex/01KXD6PA22ZZW2MJ2CH0JSKSWT) — see the study
README's "Bad-node recovery".
