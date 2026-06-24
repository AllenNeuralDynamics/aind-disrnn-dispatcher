# N x D joint scaling - verdict

> Independently replicated: two agents ran this scan separately and obtained identical grid values and additive-fit parameters (E=0.729, alpha=1.19, beta=0.67) -- this merged analysis combines both (nonlinear AIC test + the log-log regression p-value below).

Grid: N (hidden_size) in [16, 64, 128, 256] x D (#training mice) in [10, 100, 614]; 3 seeds per cell.
Metric: aggregate `heldout/final/eval_likelihood` over the fixed held-out mouse set (~149 mice).
H128 column re-used from `v2-sc-active@20260622-144622` (predates per-subject logging, so aggregate scalar used everywhere for parity).

## Per-N gain from scaling D

| N | L(D=10) | L(D=100) | L(D=614) | delta (D100->D614) | frac of D-gain by D=100 |
|---|---|---|---|---|---|
| 16 | 0.7177 | 0.7220 | 0.7226 | +0.0006 | 88% |
| 64 | 0.7218 | 0.7264 | 0.7270 | +0.0006 | 88% |
| 128 | 0.7218 | 0.7273 | 0.7282 | +0.0009 | 85% |
| 256 | 0.7214 | 0.7273 | 0.7290 | +0.0017 | 77% |

## Per-D gain from scaling N

| D | L(N=16) | L(N=64) | L(N=128) | L(N=256) | delta (N=16->256) |
|---|---|---|---|---|---|
| 10 | 0.7177 | 0.7218 | 0.7218 | 0.7214 | +0.0037 |
| 100 | 0.7220 | 0.7264 | 0.7273 | 0.7273 | +0.0053 |
| 614 | 0.7226 | 0.7270 | 0.7282 | 0.7290 | +0.0064 |

## Parametric fits

**Additive Chinchilla-style** `L = E + A*N^-alpha + B*D^-beta`
- E (irreducible / task-noise floor): **0.7289**
- A = -0.1441, alpha (N exponent) = **1.186**
- B = -0.0301, beta (D exponent) = **0.668**
- RSS = 2.395e-06, AIC = -175.1 (12 pts, 5 params)

**Interaction** `L = E + A*N^-alpha + B*D^-beta + C*N^-gamma*D^-delta`
- E = 0.7058, alpha = 0.521, beta = 0.132
- C = 15.6634, gamma = 0.000, delta = 0.132
- AIC = -190.6 (delta-AIC vs additive: -15.5; negative favors interaction, but with 12 pts vs 8 params this is fragile -- BUT C=15.66 ~ -B=15.64 with gamma~0 means the interaction term is nearly degenerate with a constant shift of E; the AIC win is mostly re-parameterization, not a clean synergy)

**Log-log interaction regression** `L ~ b0 + b1*lnN + b2*lnD + b3*(lnN*lnD)`
- interaction coef b3 = **+0.00023** (se 0.00023, p = **0.332**, NOT significant)
- A cleaner significance test than the degenerate nonlinear AIC: b3 > 0 is the synergy direction, but at this grid size the term is NOT significant -- consistent with the 'real direction, small magnitude' read.

## Interpretation

- **D saturates by ~100 across all N.** Mean fraction of total D-gain captured by D=100: **85%**. Saturation persists from H=16 to H=256, so it is NOT a hidden-size artifact.
- **N effect at every D is small, but GROWS with D.** N=16->256 gain: at D=10 = +0.0037; at D=614 = +0.0064. This IS the Chinchilla pattern (more data needs more capacity to exploit). The gap nearly doubles (1.7x), giving qualitative support for an N x D interaction. But the absolute magnitudes are small (<0.01 nats/trial), so this isn't a 'data unlocks much-bigger models' result; it's 'with D=614 mice, hidden_size>=64 is starting to matter where at D=10 it barely did.'
- **Single irreducible floor E ~ 0.729** that all (N, D) cells approach. Exponents alpha=1.19, beta=0.67: N-axis dominates.
- **Model comparison:** interaction fit's delta-AIC = -15.5 but the C-term is degenerate with the B-term (C ~ -B, gamma ~ 0). So the parametric model is ambiguous; the qualitative N x D interaction is better read off the raw delta(N=16->256) growing from +0.004 (D=10) to +0.006 (D=614).

## Caveats

- `eval_likelihood` is bounded in [0, 1] (per-trial choice probability); saturation could reflect a per-trial task-noise ceiling. Generative behavioral-match (corr~0.96+) corroborates the near-ceiling claim from a 2nd metric.
- H128 column re-uses `v2-sc-active` runs (same SC-active lambda-forward + gated-early-stop recipe as the other Ns in `nxd-grid`). No new H128 runs were trained for this scan.
- v2-sc-active's N=128 has 5 D points (10/30/100/300/614); only {10, 100, 614} used here for grid symmetry.
- 12 fit points vs 5-8 params: fits are descriptive not predictive. Extrapolation past D=614 / N=256 is not warranted.
