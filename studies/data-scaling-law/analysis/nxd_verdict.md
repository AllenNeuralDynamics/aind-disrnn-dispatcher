# Joint N×D scaling — held-out LL vs (capacity N × #training-mice D) — 2026-06-24

Tests FUTURE_DIRECTIONS §2 (Chinchilla): both single axes saturate (D-alone flattens by ~100;
N-alone flat H2→H256 at fixed small D). Do N and D scale *together* — does capacity only pay
off with more data, and vice versa (an N×D interaction)?

Grid: SC-active (session_encoding=scalar, multisubject), N∈{16,64,256} × D∈{10,100,614},
3 seeds = 27 runs (nxd-grid@20260623-102649, onprem). The N=128 row is the main-study
v2-SC-active H128 D-sweep (same recipe + heldout cohort), grafted for a 4×3 grid.

**L(N, D) — held-out-mouse eval_likelihood (mean over 3 seeds):**

| N \ D | 10 | 100 | 614 | **ΔD (10→614)** |
|---|---|---|---|---|
| 16  | 0.7177 | 0.7220 | 0.7226 | +0.0049 |
| 64  | 0.7218 | 0.7264 | 0.7270 | +0.0052 |
| 128 | 0.7218 | 0.7273 | 0.7282 | +0.0064 |
| 256 | 0.7214 | 0.7273 | 0.7290 | +0.0076 |
| **ΔN (16→256)** | +0.0037 | +0.0053 | +0.0064 | |

**Verdict: a consistent N×D interaction in the Chinchilla direction, but small and not
significant at this grid size.**
- **Both margins are monotone:** the data-gain ΔD grows with capacity (+0.0049→+0.0076 as
  N:16→256) and the capacity-gain ΔN grows with data (+0.0037→+0.0064 as D:10→614). Capacity
  only pays off when there's data to exploit, and vice versa — textbook joint scaling.
- **At D=10, LL saturates/declines past N≈64** (N256 0.7214 ≤ N64/128 0.7218 — mild overfit);
  **at D=614, LL rises monotonically with N** (0.7226→0.7290). The classic "more data needs
  more capacity" signature.
- **But the effect is weak:** the log-log interaction term (lnN·lnD) is +0.00023, **p=0.33**
  (n=12 cells) — direction right, not significant. A fully *separable* fit
  `L = E − A·N^−α − B·D^−β` already explains **R²=0.98**, with **E=0.729** (irreducible
  ceiling, ≈ the within-subject ~0.73), **α=1.19** (capacity saturates fast), **β=0.67**
  (data saturates slower). So the main effects are near-separable; the interaction is a small
  second-order correction, underpowered by the 4×3 grid.

**Reading:** joint scaling exists and points the expected way (N and D are complementary), but
like every axis in this study it's *real-but-small and fast-saturating* — held-out per-trial
choice LL is near its predictability ceiling (E≈0.729). To see strong N×D scaling you'd need a
metric with more headroom (adaptation-efficiency, OOD, lick-level — FUTURE_DIRECTIONS §1/3/7),
not a denser grid on this one.

Caveats: (1) N=128 row is a graft from a separate (same-recipe) launch; (2) grid D∈{10,100,614}
only (no 30/300), 3 N values + graft → interaction CIs are wide; (3) SC-active throughout (the
v2 arm) — an SC-off N×D grid isn't run.

Data: `nxd_result.json`; figure `fig_nxd_result.png` (LL-vs-D by N + L(N,D) heatmap). W&B group
`nxd-grid@20260623-102649` in [mice_data_scaling](https://wandb.ai/AIND-disRNN/mice_data_scaling).
