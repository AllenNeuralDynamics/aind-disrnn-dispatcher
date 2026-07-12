# RL baseline (simple, per-mouse independent fit) — verdict

Source: [cdq292n5](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/cdq292n5) (group `rl-baseline-simple@20260624-171829`). Model: `baseline_rl` / `ForagerQLearning` (L1F1_CK1 — 1 learn-rate, 1 forget-rate, 1-step choice kernel, softmax). One DE optimizer fit per held-out mouse on its own train sessions, scored on its own eval sessions. Same fixed held-out cohort (n=149) and eval sessions as the GRU.

## RL reference band

- **Pooled (trial-weighted) likelihood:** **0.7143** (matches GRU `heldout/eval_likelihood`; used as the band on Results 1, 7).
- **Per-subject mean likelihood:** **0.7211** ± 0.0052 SE, median 0.7305 (used on per-mouse panels — Results 4, 5).
- n = 149 held-out mice (1,013,989 eval trials).

Per-curriculum breakdown:

| curriculum | n | mean LL | std |
|---|---|---|---|
| Uncoupled Baiting | 71 | 0.7245 | 0.0460 |
| None | 32 | 0.6827 | 0.0884 |
| Mixed | 27 | 0.7330 | 0.0463 |
| Uncoupled Without Baiting | 17 | 0.7733 | 0.0358 |
| Coupled Baiting | 2 | 0.6125 | 0.0110 |

## Result 8 — paired GRU vs RL per held-out mouse (avg over 3 seeds)

| variant | D | n | GRU mean | RL mean | meanΔ (GRU−RL) | medianΔ | %GRU wins | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| v1 | 10 | 149 | 0.7285 | 0.7211 | +0.00738 | +0.00671 | 97% | 9.0e-26 |
| v1 | 30 | 149 | 0.7316 | 0.7211 | +0.01045 | +0.00944 | 100% | 3.4e-26 |
| v1 | 100 | 149 | 0.7328 | 0.7211 | +0.01166 | +0.01073 | 100% | 3.4e-26 |
| v1 | 300 | 149 | 0.7332 | 0.7211 | +0.01206 | +0.01150 | 100% | 3.4e-26 |
| v1 | 614 | 149 | 0.7333 | 0.7211 | +0.01214 | +0.01156 | 100% | 3.4e-26 |
| v2 | 10 | 149 | 0.7285 | 0.7211 | +0.00731 | +0.00659 | 97% | 9.8e-26 |
| v2 | 30 | 149 | 0.7315 | 0.7211 | +0.01035 | +0.00938 | 100% | 3.4e-26 |
| v2 | 100 | 149 | 0.7338 | 0.7211 | +0.01268 | +0.01173 | 100% | 3.4e-26 |
| v2 | 300 | 149 | 0.7346 | 0.7211 | +0.01342 | +0.01244 | 100% | 3.4e-26 |
| v2 | 614 | 149 | 0.7347 | 0.7211 | +0.01360 | +0.01234 | 100% | 3.4e-26 |

## Interpretation

- **GRU beats RL at every (variant, D) cell.** Even the smallest population GRU (v2, D=10) beats per-mouse RL: meanΔ=+0.00731, 97% of mice (Wilcoxon p=1e-25). A population model trained on as few as 10 other mice generalizes better to a new mouse than fitting that mouse's own data with a classical RL model — strong evidence the GRU exploits cross-mouse structure that per-mouse RL can't.
- **Large-D gain over RL.** v2 D=614 vs RL: meanΔ=+0.01360 (100% mice, p=3e-26). An order of magnitude larger than the v2−v1 SC effect (+0.0015): the population vs per-mouse cognitive-model gap is the dominant signal, not session conditioning or data scaling within the GRU.
- **SC adds a small extra margin over RL.** v2 D=614 beats RL by +0.01360; v1 D=614 by +0.01214 → v2's incremental win vs RL is +0.00146, consistent with the matched-pair v2−v1 SC result (Result 1).

## Caveats

- **The RL baseline has no D-axis.** It's a per-mouse independent fit (no cross-mouse sharing, no embeddings, no population prior). Answers "does GRU beat a stable classical RL on the same data?" (yes), not "does more mice help a population RL?" (planned hierarchical-Bayesian baseline; see `FUTURE_DIRECTIONS.md`).
- **Simple agent (L1F1_CK1):** 1 learn-rate, 1 forget-rate, 1-step choice kernel. Richer RL families (more learning/forget rates; ForagerLossCounting; ForagerCompareThreshold) may close some of the gap. Cf. `code/config/model/baseline_rl.yaml`.
- **Single optimizer seed.** Differential evolution is stable; pilot didn't need multi-seed averaging (variant notes). Add seeds [1, 2] for a tighter RL comparison.
- **Pooled vs per-subject mean differ** because mice have different trial counts. Use the trial-weighted pooled (0.7143) for GRU pooled aggregates; use the per-subject mean (0.7211) on per-mouse panels. The figures label both.
