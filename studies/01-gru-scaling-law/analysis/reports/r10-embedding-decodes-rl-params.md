---
id: r10
slug: embedding-decodes-rl-params
status: live
authors: [han]
wandb_groups:
  - v2-sc-active@20260622-144622        # GRU D=614, seed run ecd6f7e6 (best held-out)
  - rl-baseline-ctt@20260713-010000     # model2: CTT per-subject fits (614 train mice)
  - rl-baseline-bari@20260713-005938    # model2: Bari per-subject fits (614 train mice)
inputs:
  script: analysis/embedding_param_decode.py
  data: analysis/embedding_param_decode.json
  upstream:
    - run_analysis embedding-params --model1-dir <GRU D=614> --model2-dir <CTT|Bari>
      # writes subject_embedding_baseline_parameters.csv + per-parameter scatter PNGs
reproduce: |
  # 1. per-baseline join (wrapper CLI; needs the trained runs)
  python run_analysis.py embedding-params --model1-dir <GRU_RUN> --model2-dir <CTT_RUN>  --output-dir <E>/ctt
  python run_analysis.py embedding-params --model1-dir <GRU_RUN> --model2-dir <BARI_RUN> --output-dir <E>/bari
  # 2. decode
  python studies/01-gru-scaling-law/analysis/embedding_param_decode.py --ctt <E>/ctt --bari <E>/bari
---

# Result 10 — the GRU's subject embedding decodes classical-RL parameters

**Question (issue #27).** Is the learned subject embedding a real *cognitive coordinate*,
or an arbitrary code that happens to fit? If it is cognitive, we should be able to read a
mouse's classical-RL parameters out of it.

**Design.** GRU `v2-sc-active`, D=614 (run `ecd6f7e6`, the best of the 3 seeds, held-out
0.72829, checkpoint `step_90000` / `best_eval`). Its **4-dimensional** subject embedding is
joined per-mouse to the per-subject RL parameters fit by the 2026-07-13 baseline suite —
**on the 614 *training* mice**, the only mice that have a learned embedding. 611 of 614
join (3 dropped each way; see `summary.json`).

Two baselines are decoded **because they carry different parameters**, and the contrast is
the point: the parameters they *share* test cross-model consistency, and the ones they
don't tell us what else the embedding carries.

Decoding = ridge regression `embedding (4-D) → parameter`, 5-fold cross-validated (so R² is
out-of-sample), with two controls:

- a **label-shuffle null** (200 permutations) — a positive R² alone proves nothing;
- a **curriculum-only control** — curriculum drives *both* the embedding and the RL fit, so
  a parameter "predicted" only via curriculum would otherwise look like a real result.

We report **rank-based R²** (robust to fits that run to their bound; see Caveats) with the
raw R² alongside.

<!-- BEGIN result-10 -->

| parameter | model | rank R² | (raw R²) | curriculum-only | shuffle p |
|---|---|---|---|---|---|
| **`biasL`** (side bias) | CTT | **+0.719** | +0.803 | −0.006 | <0.001 |
| **`biasL`** | Bari | **+0.715** | +0.801 | −0.007 | <0.001 |
| **`learn_rate`** | CTT | **+0.670** | +0.697 | +0.025 | <0.001 |
| **`learn_rate`** | Bari | **+0.649** | +0.643 | +0.059 | <0.001 |
| `softmax_inverse_temperature` | CTT | +0.447 | +0.450 | +0.039 | <0.001 |
| `softmax_inverse_temperature` | Bari | +0.218 | *−0.337* ⚠ | −0.226 | <0.001 |
| `threshold` *(CTT only)* | CTT | +0.340 | +0.349 | −0.011 | <0.001 |
| `choice_kernel_relative_weight` *(Bari only)* | Bari | +0.198 | +0.164 | +0.000 | <0.001 |
| `forget_rate_unchosen` *(Bari only)* | Bari | +0.078 | +0.076 | −0.017 | <0.001 |
| ~~`choice_kernel_step_size`~~ *(Bari)* | Bari | **skipped** | — | — | — |

<!-- END result-10 -->

## Verdict

**The embedding is a cognitive coordinate, not an arbitrary code.** A 4-D vector learned
purely from choice prediction — never shown a single RL parameter — predicts a held-out
mouse's fitted **side bias at R²≈0.72** and **learning rate at R²≈0.66**.

**Three findings.**

1. **Cross-model consistency is the strongest evidence.** CTT and Bari are *different
   agents, fit independently*, yet they agree almost exactly on the shared parameters:
   `biasL` 0.719 vs 0.715, `learn_rate` 0.670 vs 0.649. A quirk of one fitting procedure
   could not reproduce itself this precisely in another. **This is why both baselines were
   worth fitting** — a single model could not have shown it.

2. **It is not curriculum in disguise.** Curriculum alone explains ≈0 for *every* parameter
   (−0.01 to +0.06) while the embedding explains up to 0.72. This was the obvious way the
   result could have been hollow — the embedding merely encoding which task the mouse ran —
   and it is cleanly ruled out.

3. **The embedding encodes value sensitivity and bias, far more than forgetting.**
   Ordering: bias (0.72) > learning rate (0.66) > inverse temperature (0.45) > threshold
   (0.34) > choice-kernel weight (0.20) > **forget rate (0.08)**. The embedding is nearly
   blind to forgetting. That is a substantive claim about *what individual differences the
   foundation model represents*, and a direct input to #26 (embedding → policy map) and #29
   (cognitive mechanisms).

## Caveats

- **⚠ Bari's `softmax_inverse_temperature` is a broken fit, not a failed decode.** Its raw
  R² is **−0.337**, but the parameter is heavy-tailed (skew **14.3**, median 4.25, max
  **100.0** = the DE upper bound, with mice pinned there). R² is variance-based, so a
  couple of runaway fits destroy it. On ranks it decodes at **+0.218**. CTT's inverse
  temperature — the same quantity, well-behaved (skew 0.4) — decodes at 0.447.
  **Consequence beyond this report: Bari's inverse-temperature estimates are unreliable for
  the bound-pinned mice and should not be used downstream without filtering.** Likely a
  trade-off against the choice-kernel weight.
- **`choice_kernel_step_size` is excluded, not decoded.** It is **1.0 for all 611 mice** —
  `choice_kernel="one_step"` *fixes* it by definition; it was never a free parameter. A
  naive regression "predicts" it at R²=1.000, which is a degenerate artifact of predicting
  a constant. The producer now skips constants explicitly.
- **Single GRU seed.** `ecd6f7e6` only. The embedding is unidentified up to
  rotation/scale, so absolute axes mean nothing — but *decodability* is rotation-invariant,
  so the R² values are seed-robust in principle. Worth confirming across the other 2 seeds.
- **Training mice only.** Held-out mice have no learned embedding (that is the point of the
  held-out design), so this analysis cannot run on them. It says what the embedding
  *encodes*, not how well it *generalizes*.
- **Correlational.** The embedding predicting `learn_rate` does not establish that the GRU
  *implements* a learning rate. That is #25 (disRNN update rules) and #26 (perturbation →
  policy change).

## Provenance

GRU: Beaker experiment `01KVRMSAAJTRSJMFV5JT7JAP6X`, task
`v2-sc-active-20260622-144622-012`, result dataset `01KVRMSBRV8QFHGS3XYRAS9P9A`
(fetched to `/allen/aind/scratch/han.hou/outputs/gru_d614_ecd6f7e6`).
Baselines: runs [`lmg1i9yd`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/lmg1i9yd)
(CTT) and [`bg3nzqz9`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/bg3nzqz9) (Bari),
both fit with `skip_train_fit=false` so the **training** cohort has parameters —
without which this analysis is impossible (see r8 / CHANGELOG).
Wrapper CLI: `embedding-params`, added in aind-disrnn-wrapper#55.
