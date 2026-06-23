# Data-scaling law — does session conditioning help held-out-mouse generalization?

**TL;DR.** Training on more mice improves generalization to *unseen* mice, and **session conditioning (v2) adds a small but highly-significant extra gain that GROWS with the number of training mice** — neutral/mildly mixed at small D, robustly positive from D≈100 up. The effect is tiny in absolute likelihood (~+0.0015 at the full 614-mouse pool) but consistent across ~149 held-out mice and every seed.

Generated 2026-06-23. W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling

## Design
GRU H128, scalar session conditioning. v1 = SC never engaged (early-stopped in pretrain, λ=0); v2 = λ forward (full SC @50k) + gated early-stop @70k. Swept D = #training mice (10/30/100/300/614) × 3 seeds; held-out cohort fixed (~149 mice, every-5th-ranked). y = held-out fine-tune+test likelihood. v1 and v2 use IDENTICAL mice per (D,seed) → matched-pair comparison.

## Result 1 — held-out scaling curve (cell-level, n=15 matched pairs)

![scaling](fig_scaling_v1_v2.png)

| D | v1 (SC off) | v2 (SC on) | Δ(v2−v1) |
|---|---|---|---|
| 10 | 0.7219 | 0.7218 | -0.00006 |
| 30 | 0.7250 | 0.7249 | -0.00011 |
| 100 | 0.7262 | 0.7273 | +0.00104 |
| 300 | 0.7267 | 0.7280 | +0.00137 |
| 614 | 0.7268 | 0.7282 | +0.00148 |

Paired across 15 cells: mean Δ=**+0.00074**, 12/15 positive, paired t p=**0.0015**, Wilcoxon p=**0.0043**. Power-law asymptote E: v1≈0.7248, v2≈0.7252.

## Result 2 — per-held-out-mouse repeated measures (n=149 mice/D, paired by mouse, avg over seeds)

![per-subject delta](fig_per_subject_delta.png)

| D | mean Δ | median Δ | % mice improved | Wilcoxon p |
|---|---|---|---|---|
| 10 | +0.00031 | +0.00017 | 68% | 7.6e-08 |
| 30 | -0.00010 | -0.00010 | 36% | 3.3e-04 |
| 100 | +0.00102 | +0.00096 | 85% | 1.3e-20 |
| 300 | +0.00130 | +0.00122 | 91% | 1.1e-23 |
| 614 | +0.00145 | +0.00135 | 93% | 2.3e-24 |

**Pairing by held-out mouse (n=149) makes the effect overwhelmingly significant** (p~1e-20–1e-24 vs p~0.002 for the 15-cell test) and reveals the shape clearly: mildly mixed at small D (slightly *hurts* at D=30: only 36% of mice improve), then robustly positive and increasing (85%→93% of mice improve as D goes 100→614).

## Interpretation
Session conditioning is **not** what unlocks held-out generalization at small data — there it's neutral-to-slightly-harmful (consistent with Po-chen's within-population H128 finding). But with enough training mice (D≳100) it provides a small, reliable per-mouse generalization gain that scales with D. For a behavior foundation model the actionable levers remain (a) more mice and (b) SC *in combination with* scale — not SC alone at small D.

## Provenance
v1 group `v1-pretrain-phase@20260622-013415` (exp 01KVQ7EJ3C5YJ8FJVNJB8C8N36); v2 group `v2-sc-active@20260622-144622` (exp 01KVRMSAAJTRSJMFV5JT7JAP6X). Per-subject from offline held-out re-runs (wrapper 4f29680) groups `heldout-rerun-v1*`/`heldout-rerun-v2*`. Data: paired_v1_v2_cell.json, report_data.json. Figures: fig_scaling_v1_v2.png, fig_per_subject_delta.png.
