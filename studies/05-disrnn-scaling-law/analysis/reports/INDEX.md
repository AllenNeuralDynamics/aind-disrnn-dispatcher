# 05-disrnn-scaling-law — reports

One producer for both: `analysis/scaling_report.py` (reads the committed `analysis/grid.csv`, writes
`summary.{json,csv}` + the three figures, regenerates the `<!-- BEGIN result-N -->` blocks).
Regenerate everything with `make -C studies/05-disrnn-scaling-law`.

| report | question | verdict |
|---|---|---|
| [r1 — held-out scaling](r1-heldout-scaling.md) | Does the disRNN transfer better with more mice? | **No.** It peaks at D≈100 (0.7174) then **declines** (0.7154 at D=614) — not undertraining. It sits ~0.010 below the GRU at *every* D and, at the full cohort, below a per-mouse RL baseline. |
| [r2 — sparsity & the multiplier](r2-sparsity-and-multiplier.md) | Does study 03's D=100 verdict hold at D=614? | **Half of it.** The multiplier still closes the gate monotonically, and "more mice ⇒ less sparse" is confirmed. But study 03's headline — *sparsity is free* — **breaks**: at D=614 sparsifying costs ~0.004 held-out, half the disRNN's gap to the GRU. |
| r3 — subject capacity | Is per-subject capacity the transfer cap? | ⏳ pending — `subject-capacity` (18 tasks) still running; no held-out numbers written yet. |
| [r4 — generative match](r4-generative-behavioral-match.md) | Does the disRNN *behave* like a mouse? | **Less than the GRU does, at every D.** History-curve correlation trails by 0.02–0.03 — **10–20× the seed noise**. RMSE is comparable, so it gets the average switch level right and the *shape* wrong. The 2nd-order test discriminates where likelihood could not. |

## Metric caveat (carry into every report)

Bottleneck openness is **`total_openness` = Σ(1−σ)**, *never* `n_eff_open_frac` — the latter is
scale-invariant and reads high even for a fully shut bottleneck (it mis-ranked 19/43 runs in study
03). See [`../../03-disrnn-beta-scan/analysis/provenance/metric_caveat.md`](../../../03-disrnn-beta-scan/analysis/provenance/metric_caveat.md).

## Noise caveat (learned the hard way here)

**Held-out likelihood and bottleneck openness have wildly different seed stability.** Measured on all four runs at the same config (D=614, mult=2, β=1e-3; wave 1 seeds 0/1/2 plus
wave 2's seed 42 — wave 2 never set `seed`, so it ran at Hydra's default 42, an independent 4th seed):

- held-out LL: SD **0.00046** (seeds 0/1/2/42: 0.7157 / 0.7153 / 0.7153 / 0.7163) — steady
- interaction openness: SD **0.384** (1.136 / 0.467 / 0.474) — enormous

So a ~0.004 held-out effect is real (8.4× the noise), while a ~0.2 openness difference is **not**. An
earlier draft of this study read a 3-seed openness *mean* as evidence of non-monotonicity vs D; that
was noise and the claim was withdrawn. **Openness claims need seed replication.**
