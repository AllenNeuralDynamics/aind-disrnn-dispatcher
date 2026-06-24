# Data-scaling law — does training on more mice (and session conditioning) help held-out-mouse generalization?

**TL;DR.** Training on more mice improves prediction of *unseen* mice, but the gain **saturates by ~100 mice** and the absolute ceiling is low — per-trial L/R choice likelihood is near a **predictability ceiling**. Session conditioning (v2) is neutral-to-slightly-negative at small D, then adds a **small, highly-significant gain that grows with D** (robust from D≈100); ~¾ of that gain persists on mature-only sessions, so it's mostly *not* a curriculum-stage artifact. The population-mean ("average mouse") already predicts a new mouse to within ~0.3% of full adaptation, so per-mouse few-shot adaptation is **not** where scale pays off. Effects are tiny per-trial but, being consistent across ~149 mice, are real evidence (a +0.001 per-trial-normalized LL ≈ ~2× per-session likelihood ratio). **Verdict: this metric does not validate "big data ⇒ materially better foundation model"; the headroom bets are richer outputs (3-way ignore, lick/RT) and OOD transfer.**

Generated 2026-06-23 · W&B: https://wandb.ai/AIND-disRNN/mice_data_scaling

## Design
GRU H128, scalar session conditioning. **v1** = SC never engaged (early-stopped in pretrain, λ=0); **v2** = λ-forward (full SC @50k) + gated early-stop @70k. Swept **D = #training mice** (10/30/100/300/614) × 3 seeds; held-out cohort **fixed** (~149 mice). v1 and v2 use the **identical** mice per (D, seed) → matched-pair comparison. y = held-out fine-tune+test likelihood (per-trial normalized).

## Result 1 — held-out scaling curve (cell-level, n=15 matched pairs)
![scaling](fig_scaling_v1_v2.png)

| D | v1 (SC off) | v2 (SC on) | Δ(v2−v1) |
|---|---|---|---|
| 10 | 0.7219 | 0.7218 | −0.00006 |
| 30 | 0.7250 | 0.7249 | −0.00011 |
| 100 | 0.7262 | 0.7273 | +0.00104 |
| 300 | 0.7267 | 0.7280 | +0.00137 |
| 614 | 0.7268 | 0.7282 | +0.00148 |

Paired across 15 cells: mean Δ=**+0.00074**, 12/15 positive, paired t p=**0.0015**, Wilcoxon p=**0.0043**. Curve is saturating over this D range.

## Result 2 — per-held-out-mouse repeated measures (n=149 mice/D, paired by mouse)
![per-subject delta](fig_per_subject_delta.png)

| D | mean Δ | median Δ | % mice improved | Wilcoxon p |
|---|---|---|---|---|
| 10 | −0.00007 | −0.00008 | 34% | 3.2e-06 |
| 30 | −0.00010 | −0.00010 | 36% | 3.3e-04 |
| 100 | +0.00102 | +0.00096 | 85% | 1.3e-20 |
| 300 | +0.00135 | +0.00124 | 93% | 7.5e-24 |
| 614 | +0.00146 | +0.00138 | 95% | 1.5e-24 |

Per-mouse pairing makes the effect overwhelmingly significant (p~1e-20–1e-24) and shows the shape: **neutral-to-slightly-negative at small D** (D=10: 34% improve), then robustly positive and increasing (85%→95% as D 100→614). The small-D per-mouse Δ matches the independent cell-level aggregate (both ≈−0.0001 at D=10) — a consistency check.

> **Dedup note.** Offline per-subject re-runs had duplicate runs for 10/30 (variant,ratio,seed) cells (validation + mass-launch + BLAS retries); `build_report.py` keeps one run/cell. This corrected D=10 from a spurious +0.00031/68% (double-counted v1 seed-0) to −0.00007/34%; large-D unchanged. The cell-level test (Result 1) was never affected.

## Result 3 — bootstrap CIs on the scaling shape (resample 149 held-out mice ×1000)
Per-mouse-mean LL (equal-weight; differs from the trial-weighted Result 1 levels). Within-cohort increments are tight even though absolute per-D levels aren't (mice vary in predictability).

| quantity | v1 | v2 |
|---|---|---|
| frac of total gain by D=100 | 0.90 [0.89, 0.91] | 0.85 [0.84, 0.87] |
| late gain D=100→614 | +0.00049 [+0.00042, +0.00056] | +0.00092 [+0.00084, +0.00100] |

Both late-gain CIs **exclude 0** → not perfectly saturated; a small real slope persists (≈2× larger under SC). A saturating fit `L = L∞ − A·D^−α` (seed-averaged) gives **v1: L∞=0.727, α=0.88** (asymptote essentially reached by D=614) and **v2: L∞=0.729, α=0.52** — SC's shallower exponent means it keeps rising, with the asymptote ~+0.001 above the observed D=614, echoing v2's ~2× late slope. **~85–90% of the data benefit is captured by ~100 mice.** The residual is small per-trial but statistically real and, per the per-session compounding (below), genuine evidence — just headroom-poor on this metric.

## Result 4 — zero-shot vs adapted held-out generalization
![zero-shot vs adapted](fig_zeroshot_vs_d.png)

Zero-shot = held-out mouse assigned the **population-mean embedding**, no adaptation. Adapted = embedding fine-tuned on ~half its sessions (test on the other half). Per-mouse means.

| D | v1 zero | v1 adapt | gap | v2 zero | v2 adapt | gap |
|---|---|---|---|---|---|---|
| 10 | 0.7264 | 0.7285 | +0.0021 | 0.7264 | 0.7285 | +0.0021 |
| 100 | 0.7309 | 0.7328 | +0.0019 | 0.7319 | 0.7338 | +0.0019 |
| 614 | 0.7315 | 0.7333 | +0.0018 | 0.7331 | 0.7347 | +0.0016 |

- **Adaptation buys ~+0.002 and the gap is flat across D** — the "average mouse" already predicts a new mouse to within ~0.3% of full adaptation. Subject-specific adaptation barely matters ⇒ few-shot efficiency is unlikely to be the scaling win.
- **Zero-shot scales with D** (v1 +0.0051, v2 +0.0068 over D=10→614) but **saturates ~D=100**, same shape as adapted.
- **SC's large-D edge shows even at zero-shot** (v2>v1 ~+0.0016 at D=614) — the frozen shared session-conditioning generalizes better when trained on more mice.

## Result 5 — few-shot adaptation curve (K = # new-mouse sessions used to adapt)
![few-shot curve](fig_fewshot_curve.png)

| var | D | K=0 (zero) | K=1 | K=4 | K=full(~8) |
|---|---|---|---|---|---|
| v1 | 10 | 0.7264 | 0.7047 | 0.7252 | 0.7285 |
| v1 | 614 | 0.7315 | 0.7107 | 0.7307 | 0.7333 |
| v2 | 614 | 0.7331 | 0.7153 | 0.7326 | 0.7347 |
(all 10 cells show the same shape)

**Non-monotonic:** K=1 craters LL by ~−0.02 (a 4-dim embedding overfits one session at 500 steps/lr=1e-3), recovers to ≈zero-shot by K=4, only exceeds zero-shot at K=full (+~0.002). The dip is **D- and variant-independent** (protocol property, not scale). Caveat: largely a **protocol artifact** (no L2-to-mean / early-stop / lr-scaling for small K) — a tuned few-shot likely wouldn't crash. *(Mature-only few-shot in progress to test whether the crash is also driven by adapting on naive early-stage sessions.)*

## Result 6 — SC-stage verdict (mature-only eval)
Tests "is SC's benefit just accounting for curriculum/early-stage heterogeneity?" Re-ran held-out *adapted* on **mature sessions only** (STAGE_FINAL/GRADUATED) using the same all-stage-trained checkpoints (cohort 149→117; mature LL higher, ~0.745, as mature behavior is more predictable).

| D | Δ(v2−v1) all-stage | Δ mature-only |
|---|---|---|
| 100 | +0.00102 | +0.00073 |
| 300 | +0.00135 | +0.00106 |
| 614 | +0.00146 | +0.00116 |

Large-D mean v2−v1: **+0.00128 (all-stage) → +0.00098 (mature) = ~23% shrinkage** (mature still p~1e-15). **So ~¼ of SC's benefit was the early-stage heterogeneity (the design rationale was partly real), but ~¾ persists on mature animals → SC mostly captures general session structure (within-mature drift / within-session non-stationarity), not just training stage.** (Eval-level test; models still trained all-stage. A definitive "retrain mature-only" test is deprioritized given this.)

## Result 7 — N × D joint scaling grid (Chinchilla-style)
![N x D joint scaling](fig_nxd_scaling.png)

*Heatmap and paired slices through the N×D grid. D saturates by ~100 mice at each hidden size, while the fixed-D N gain grows modestly from D=10 to D=614.*

Grid: N (hidden_size) ∈ {16, 64, 128, 256} × D ∈ {10, 100, 614} × 3 seeds (12 (N,D) cells). H128 column re-used from `v2-sc-active`. Metric: aggregate `heldout/final/eval_likelihood` across the same fixed held-out mouse set (~149 mice).

Mean L grid (held-out eval likelihood):

| N | D=10 | D=100 | D=614 | Δ (D100→D614) | frac of D-gain by D=100 |
|---|---|---|---|---|---|
| 16  | 0.7177 | 0.7220 | 0.7226 | +0.0006 | 88% |
| 64  | 0.7218 | 0.7264 | 0.7270 | +0.0006 | 88% |
| 128 | 0.7218 | 0.7273 | 0.7282 | +0.0009 | 85% |
| 256 | 0.7214 | 0.7273 | 0.7290 | +0.0017 | 77% |

- *D saturates by ~100 across every N tested* (mean 85% of D-gain captured by D=100). Saturation is *not* a hidden-size artifact — it persists from H=16 to H=256.
- *N-axis gain at fixed D grows weakly with D* (Chinchilla-style interaction). N=16→256 gain: +0.0037 at D=10, +0.0064 at D=614 (×1.7). Qualitative support for an N×D synergy, but absolute magnitudes are small (<0.01 nats/trial).
- Additive fit `L = E + A·N^{-α} + B·D^{-β}`: E≈0.729 (single irreducible floor), α≈1.19 (N), β≈0.67 (D); N-axis dominates within this grid.
- Interaction-term fit improves AIC by 15.5 *but the C-term is degenerate with the B-term* (C ≈ −B, γ ≈ 0), so the AIC win is mostly re-parameterization. The qualitative N×D interaction is better read off the raw Δ(N=16→256) row.
- *Verdict*: same predictability ceiling story as Result 1; with D=614 mice, hidden_size ≥ 64 is starting to matter where at D=10 it barely did, but the headroom is small. See `nxd_scaling_verdict.md` for the fit details.

## On effect sizes (Kevin Miller)
LL is per-trial-normalized (NL = exp(mean_t log p_t)). A *consistent* Δ=+0.001 ≈ +0.0014 nats/trial → ~0.7 nats over a ~500-trial session → **~2× per-session likelihood ratio**, compounding across sessions/mice. So the small SC / data-scaling deltas are genuine model evidence (per-mouse pairing p~1e-24 confirms), not noise — even though the metric is headroom-poor.

## Verdict
On **per-trial choice likelihood**, the system is **near a predictability ceiling**: a new mouse is predicted to ~99.7% of its adapted likelihood from the population mean; data-scaling rises fast then saturates by ~100 mice; per-mouse adaptation adds ~+0.002 (flat in D); SC adds a small, real, mostly-not-stage gain that grows with D. The **N×D joint scan (Result 7)** confirms D saturates at every N tested and the N-axis adds only a small (+0.004 to +0.006) gain that mildly grows with D — a weak Chinchilla-style interaction within this metric. None of this *invalidates* the foundation-model idea — the effects are real and compound — but this **metric/task lacks the headroom** to demonstrate big-data scaling. The tests that could: **headroom-ier targets** — 3-way output incl. ignored trials, lick/RT modeling, OOD task/rig transfer, and **fair-comparison RL baselines** (hierarchical Bayesian fit; see `FUTURE_DIRECTIONS.md`).

## Status (2026-06-23)
Done: Results 1–6 + bootstrap + generative-validation + **N×D joint scan**. In progress: mature-only few-shot (crash test), mass-generative behavioral-match-vs-D. Deferred: mature-only retrain (B); regularized few-shot.

## Provenance
Training variants: `v1-pretrain-phase@20260622-013415` (exp 01KVQ7EJ3C5YJ8FJVNJB8C8N36), `v2-sc-active@20260622-144622` (exp 01KVRMSAAJTRSJMFV5JT7JAP6X), `nxd-grid@20260623-102649`. Offline analyses (wrapper 4f29680 / few-shot knob bb4b052), one run/cell deduped: `heldout-rerun-*` (adapted), `heldout-zeroshot-*` (zero-shot), `heldout-rerun-*-fewshot-k*`, `heldout-rerun-*-mature2@` (mature), `generative-*`. Artifacts in `analysis/` (`*.json`, `fig_*.png`). Report run: https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/0fhvwwfu
