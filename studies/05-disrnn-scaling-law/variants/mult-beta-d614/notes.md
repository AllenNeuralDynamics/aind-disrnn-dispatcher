# Variant: mult-beta-d614 — does study 03's verdict survive at the full cohort?

**What differs.** Study 03's `updnet-ratio-100mice` grid, re-run at **D≈614** instead of D=100:
`update_net_latent_penalty_multiplier ∈ {1,2,5,10}` × base `β ∈ {3e-4, 1e-3, 3e-3}` = **12 tasks**.
The two nuisance axes of study 03 are dropped: `lr` pinned to 1e-3 (both of 03's NaN divergences
were at 5e-3) and `seed` pinned to 0 (this is a mechanism check, not an effect-size estimate — the
D=614 cell of [`dscan-mult2`](../dscan-mult2/notes.md) supplies 3 seeds at mult=2). Everything else
is byte-identical to study 03, so the grids compare cell-by-cell with exactly one axis moved.

**Why it exists (issue #16, need 3).** Study 03's conclusions — multiplier monotonically closes the
interaction bottleneck; the model compensates by opening other gates; held-out transfer is flat
across the multiplier — are what justify picking mult=2 for the scaling curve. But they were
established at D=100, and study 03's *own motivating premise* is that the interaction bottleneck
fails to sparsify **when many mice are trained together**. D=100 may simply be too few mice for that
failure to appear. This grid tests the premise at the cohort size where it was supposed to bite.

**Three claims to re-test at D=614:**
1. **Monotonicity.** Interaction (update←latent) openness Σ(1−σ) falls monotonically with the
   multiplier. At D=100: 3.11 → 1.16 → 0.11 → 0.00 (mult 1→2→5→10) at β=3e-4.
2. **Compensation.** As update←latent closes, `update←subject` and `choice←latent` open, the
   recurrent `latent` closes as collateral, `update←obs` stays most open, `choice←subject` stays shut.
3. **Free sparsity.** Held-out transfer is flat across the multiplier (~0.008 LL full range at
   D=100), with what little variation there is tracking base β, not the multiplier.

If (1) and (3) hold at D=614, the mult=2 operating point — and the interpretability read-out at
scale — is safe. If openness at a given (mult, β) is systematically *higher* at D=614 than at
D=100, that is the "more mice → less sparse" effect the multiplier was invented to fix, and it
would say the multiplier needs to be **scaled with D**, not held fixed.

**Watch.** mult=10 was NaN-prone in study 03 (deterministic at lr=5e-3/seed=0). We avoid that lr,
but a mult=10 divergence at D=614 is itself a reportable result — record it, do not silently retry
it into existence.

**Metric caveat.** Openness = **Σ(1−σ)** (`total_openness`), never `n_eff_open_frac` — see
`03-disrnn-beta-scan/analysis/provenance/metric_caveat.md`.

**Cost.** 12 tasks × ~18 h ≈ **216 GPU-h**, low-preemptible on g6e L40S.

**Status.** ⏳ running 12/12 — launched 2026-07-13 00:35 PT, group `mult-beta-d614@20260713-003501`,
Beaker [`01KXD6DCD9VGY3G6D3M0JWPB7X`](https://beaker.org/ex/01KXD6DCD9VGY3G6D3M0JWPB7X). Six cells
— (mult=1, all three β), (mult=2, β=3e-4), (mult=2, β=3e-3), (mult=5, β=3e-4) — died pre-start on
the bad node `aidc-h200-prd2` and were re-submitted verbatim (same run ids + group) off that
cluster as [`01KXD6PBQ8CDVG7RF8S7DJ64MF`](https://beaker.org/ex/01KXD6PBQ8CDVG7RF8S7DJ64MF) — see
the study README's "Bad-node recovery".
