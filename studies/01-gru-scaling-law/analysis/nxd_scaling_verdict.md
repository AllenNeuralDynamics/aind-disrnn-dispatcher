# N x D joint scaling - verdict

> Independently replicated: two agents ran this scan separately and obtained identical grid values and additive-fit parameters (E=0.729, alpha=1.19, beta=0.67) before the D=30 gap-fill. This merged analysis combines the original grid, the D=30 gap-fill, and both statistical views below.

Grid: N (hidden_size) in [16, 64, 128, 256, 512] x D (#training mice) in [10, 30, 100, 614]; 3 seeds per cell.
Metric: aggregate `heldout/final/eval_likelihood` over the fixed held-out mouse set (~149 mice).
H128 column re-used from `v2-sc-active@20260622-144622` (predates per-subject logging, so aggregate scalar used everywhere for parity).

## Per-N gain from scaling D

| N | L(D=10) | L(D=30) | L(D=100) | L(D=614) | delta (D100->D614) | frac of D-gain by D=100 |
|---|---|---|---|---|---|---|
| 16 | 0.7177 | 0.7200 | 0.7220 | 0.7226 | +0.0006 | 88% |
| 64 | 0.7218 | 0.7247 | 0.7264 | 0.7270 | +0.0006 | 88% |
| 128 | 0.7218 | 0.7249 | 0.7273 | 0.7282 | +0.0009 | 85% |
| 256 | 0.7214 | 0.7251 | 0.7273 | 0.7290 | +0.0017 | 77% |
| 512 | 0.7213 | 0.7253 | 0.7278 | n/a | n/a (no D=614) | n/a |

## Per-D gain from scaling N

| D | L(N=16) | L(N=64) | L(N=128) | L(N=256) | L(N=512) | delta (N=16->512) |
|---|---|---|---|---|---|---|
| 10 | 0.7177 | 0.7218 | 0.7218 | 0.7214 | 0.7213 | +0.0036 |
| 30 | 0.7200 | 0.7247 | 0.7249 | 0.7251 | 0.7253 | +0.0053 |
| 100 | 0.7220 | 0.7264 | 0.7273 | 0.7273 | 0.7278 | +0.0058 |
| 614 | 0.7226 | 0.7270 | 0.7282 | 0.7290 | n/a | n/a (N=512 has no D=614 cell) |

## Parametric fits

**Additive Chinchilla-style** `L = E + A*N^-alpha + B*D^-beta`
- E (irreducible / task-noise floor): **0.7290**
- A = -0.2351, alpha (N exponent) = **1.374**
- B = -0.0279, beta (D exponent) = **0.604**
- RSS = 3.768e-06, AIC = -283.2 (19 pts, 5 params)

**Interaction** `L = E + A*N^-alpha + B*D^-beta + C*N^-gamma*D^-delta`
- E = 0.7300, alpha = 0.756, beta = 0.373
- C = 18.4601, gamma = 0.000, delta = 0.373
- AIC = -304.7 (delta-AIC vs additive: -21.5; negative favors interaction, but with 19 pts vs 8 params this is fragile -- BUT C=18.46 ~ -B=18.47 with gamma~0 means the interaction term is nearly degenerate with a constant shift of E; the AIC win is mostly re-parameterization, not a clean synergy)

**Log-log interaction regression** `L ~ b0 + b1*lnN + b2*lnD + b3*(lnN*lnD)`
- interaction coef b3 = **+0.00038** (se 0.00018, p = **0.058**, NOT significant)
- A cleaner significance test than the degenerate nonlinear AIC: b3 > 0 is the synergy direction, but at this grid size the term is NOT significant -- consistent with the 'real direction, small magnitude' read.

## Interpretation

- **D saturates by ~100 across all N.** Mean fraction of total D-gain captured by D=100: **85%**. Saturation persists from H=16 to H=256, so it is NOT a hidden-size artifact.
- **N effect at every D is small, but grows with D.** N=16->512 gain at D=10 = +0.0036 (N=512 has no D=614 cell -- host-RAM OOM, see variants/nxd-h512/notes.md -- so the D=614 comparison below uses N=16->256, the largest N that has one). N=16->256 gain at D=614 = +0.0064. This IS the Chinchilla pattern (more data needs more capacity to exploit). The gap grows (1.8x), giving qualitative support for an N x D interaction. But the absolute magnitudes are small (<0.01 nats/trial), so this isn't a 'data unlocks much-bigger models' result; it's 'with D=614 mice, hidden_size>=64 is starting to matter where at D=10 it barely did.'
- **Single irreducible floor E ~ 0.729** that all (N, D) cells approach. Exponents alpha=1.37, beta=0.60: N-axis dominates.
- **Model comparison:** interaction fit's delta-AIC = -21.5 but the C-term is degenerate with the B-term (C ~ -B, gamma ~ 0). So the parametric model is ambiguous; the qualitative N x D interaction is better read off the raw delta(N=16->256) growing from +0.004 (D=10) to +0.006 (D=614).

## Caveats

- `eval_likelihood` is bounded in [0, 1] (per-trial choice probability); saturation could reflect a per-trial task-noise ceiling. Generative behavioral-match (corr~0.96+) corroborates the near-ceiling claim from a 2nd metric.
- H128 column re-uses `v2-sc-active` runs (same SC-active lambda-forward + gated-early-stop recipe as the other Ns in `nxd-grid`). No new H128 runs were trained for this scan.
- v2-sc-active's N=128 has 5 D points (10/30/100/300/614); only {10, 30, 100, 614} used here for grid symmetry.
- 19 fit points (20 rectangular grid cells, 1 unattainable -- see variants/nxd-h512/notes.md) vs 5-8 params: fits are descriptive not predictive. Extrapolation past D=614 / N=512 is not warranted.
