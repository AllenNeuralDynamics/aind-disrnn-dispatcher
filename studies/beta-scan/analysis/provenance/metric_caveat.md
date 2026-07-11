# Correctness caveat: openness metric (`total_openness` vs `n_eff_open_frac`)

**Carry this into every reading of the β-scan results.** An earlier version of
this analysis headlined `n_eff_open_frac` and reached a *wrong* conclusion
(a U-shaped multiplier axis and an inverted β ranking). The reports and figures
here use `total_openness` and are correct. This note records the trap so it is
not re-set.

## The two metrics

For a bottleneck family, let `σ_i ∈ (0,1]` be the learned per-channel noise gate
(small σ = OPEN, σ→1 = CLOSED), and `w_i = clip(1−σ_i, 0, 1)` the per-channel
openness.

- **`total_openness` = Σ w_i** — absolute open capacity, in nats. Reads ~0 when a
  bottleneck is fully closed. **This is what every conclusion uses.**
- **`n_eff_open_frac` = (Σ w_i)² / (N · Σ w_i²)** — the *normalized participation
  ratio*: the effective fraction of channels carrying the openness, ∈ [1/N, 1].
  Scale-invariant (multiply all w_i by any constant and it is unchanged).

## Why `n_eff_open_frac` misleads here

Because it is scale-invariant, `n_eff_open_frac` characterizes only *how the
openness is distributed across channels*, not *how much openness exists*. When a
bottleneck is fully shut (all σ_i ≈ 0.999, all w_i ≈ 1e-3), the residual noise-floor
weights are roughly uniform, so `n_eff_open_frac` reports a **high** value —
exactly backwards from "this gate is closed."

Concrete example from the grid (`update_net_latent`, N=25):

| run | regime | total_openness Σ(1−σ) | n_eff_open_frac | truth |
|---|---|---|---|---|
| `45646c46` | mult=2, β=3e-3, lr=5e-3, s=1 | **0.003** (min_σ 0.999) | 0.257 (grid **max**) | fully CLOSED |
| `3dcb9217` | mult=2, β=3e-4, lr=1e-3, s=1 | **0.58** (min_σ 0.43) | 0.041 | 1 genuinely OPEN channel |

`n_eff_open_frac` ranks `45646c46` as the *most open* interaction bottleneck in
the grid; `total_openness` (and the actual σ values) show it is the *most closed*.
**19 of 43** runs had `n_eff_open_frac > 0.12` while `total_openness < 0.1`.

## Effect on conclusions

- **Multiplier axis:** `n_eff_open_frac` produced a spurious U-shape (dipping then
  rising with the multiplier); `total_openness` shows the true **monotone closure**
  (r1).
- **β ranking:** by `total_openness`, weak β=3e-4 is most open (mean 1.17), β=1e-3
  intermediate (0.62), β=3e-3 most closed (0.002). `n_eff_open_frac` inverted this,
  ranking β=3e-3 highest (0.138).

## What is kept

The full threshold-free suite — including `n_eff_open_frac`, `n_eff_open`,
`frac_open_s{0p03..0p97}`, `mean/min_sigma`, `sigma_p10/median/p90` — is retained
per run in `analysis/beta_scan_final_grid.csv` for reference and re-analysis. Only
the *headline metric* choice is constrained: use `total_openness`.
