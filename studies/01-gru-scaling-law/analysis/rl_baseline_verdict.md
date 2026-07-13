# RL baseline (simple, per-mouse independent fit) — verdict

Source: [bg3nzqz9](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/bg3nzqz9) (group `None`). Model: `baseline_rl` / `ForagerQLearning` (L1F1_CK1 — 1 learn-rate, 1 forget-rate, 1-step choice kernel, softmax). One DE optimizer fit per held-out mouse on its own train sessions, scored on its own eval sessions. Same fixed held-out cohort (n=149) and eval sessions as the GRU.

## RL reference band

- **Pooled (trial-weighted) likelihood:** **0.7149** (matches GRU `heldout/eval_likelihood`; used as the band on Results 1, 7).
- **Per-subject mean likelihood:** **0.7217** ± 0.0052 SE, median 0.7311 (used on per-mouse panels — Results 4, 5).
- n = 149 held-out mice (1,013,172 eval trials).

Per-curriculum breakdown:

| curriculum | n | mean LL | std |
|---|---|---|---|
| Uncoupled Baiting | 71 | 0.7257 | 0.0467 |
| Mixed | 27 | 0.7333 | 0.0457 |
| Uncoupled Without Baiting | 17 | 0.7732 | 0.0358 |
| Coupled Baiting | 2 | 0.6125 | 0.0110 |

## Result 8 — paired GRU vs RL per held-out mouse (avg over 3 seeds)

| variant | D | n | GRU mean | RL mean | meanΔ (GRU−RL) | medianΔ | %GRU wins | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| v1 | 10 | 146 | 0.7288 | 0.7215 | +0.00732 | +0.00656 | 97% | 2.9e-25 |
| v1 | 30 | 146 | 0.7318 | 0.7215 | +0.01037 | +0.00934 | 100% | 1.0e-25 |
| v1 | 100 | 146 | 0.7330 | 0.7215 | +0.01157 | +0.01071 | 100% | 1.0e-25 |
| v1 | 300 | 146 | 0.7334 | 0.7215 | +0.01198 | +0.01118 | 100% | 1.0e-25 |
| v1 | 614 | 146 | 0.7335 | 0.7215 | +0.01206 | +0.01125 | 100% | 1.0e-25 |
| v2 | 10 | 146 | 0.7287 | 0.7215 | +0.00725 | +0.00657 | 97% | 3.1e-25 |
| v2 | 30 | 146 | 0.7317 | 0.7215 | +0.01028 | +0.00935 | 100% | 1.0e-25 |
| v2 | 100 | 146 | 0.7341 | 0.7215 | +0.01260 | +0.01139 | 100% | 1.0e-25 |
| v2 | 300 | 146 | 0.7348 | 0.7215 | +0.01333 | +0.01206 | 100% | 1.0e-25 |
| v2 | 614 | 146 | 0.7350 | 0.7215 | +0.01352 | +0.01227 | 100% | 1.0e-25 |

## Interpretation

- **GRU beats RL at every (variant, D) cell.** Even the smallest population GRU (v2, D=10) beats per-mouse RL: meanΔ=+0.00725, 97% of mice (Wilcoxon p=3e-25). A population model trained on as few as 10 other mice generalizes better to a new mouse than fitting that mouse's own data with a classical RL model — strong evidence the GRU exploits cross-mouse structure that per-mouse RL can't.
- **Large-D gain over RL.** v2 D=614 vs RL: meanΔ=+0.01352 (100% mice, p=1e-25). An order of magnitude larger than the v2−v1 SC effect (+0.0015): the population vs per-mouse cognitive-model gap is the dominant signal, not session conditioning or data scaling within the GRU.
- **SC adds a small extra margin over RL.** v2 D=614 beats RL by +0.01352; v1 D=614 by +0.01206 → v2's incremental win vs RL is +0.00147, consistent with the matched-pair v2−v1 SC result (Result 1).

## Caveats

- **The RL baseline has no D-axis.** It's a per-mouse independent fit (no cross-mouse sharing, no embeddings, no population prior). Answers "does GRU beat a stable classical RL on the same data?" (yes), not "does more mice help a population RL?" (planned hierarchical-Bayesian baseline; see `FUTURE_DIRECTIONS.md`).
- **Simple agent (L1F1_CK1):** 1 learn-rate, 1 forget-rate, 1-step choice kernel. Richer RL families (more learning/forget rates; ForagerLossCounting; ForagerCompareThreshold) may close some of the gap. Cf. `code/config/model/baseline_rl.yaml`.
- **Single optimizer seed.** Differential evolution is stable; pilot didn't need multi-seed averaging (variant notes). Add seeds [1, 2] for a tighter RL comparison.
- **Pooled vs per-subject mean differ** because mice have different trial counts. Use the trial-weighted pooled (0.7143) for GRU pooled aggregates; use the per-subject mean (0.7211) on per-mouse panels. The figures label both.
