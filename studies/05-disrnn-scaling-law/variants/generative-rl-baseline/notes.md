# Variant: generative-rl-baseline — do the classical RL models behave more like a mouse than the disRNN?

**What it is.** Not a training run. Roll each of r1's three per-mouse RL baselines out as a
generative agent — same curves, same code path, same tasks as [`generative-dscan`](../generative-dscan/notes.md) —
so [r4](../../analysis/reports/r4-generative-behavioral-match.md) can carry RL reference lines
alongside the disRNN and the GRU.

**Why it matters.** [r1](../../analysis/reports/r1-heldout-scaling.md) already shows the disRNN
*loses to compare-to-threshold on held-out likelihood at D=614* (0.7154 vs 0.7170) — a baseline the
GRU beats by +0.0098. [r4](../../analysis/reports/r4-generative-behavioral-match.md) then shows the
disRNN's generative behavior trails the GRU's at every D. The question that sets up has never been
asked: **does a 4-parameter RL model also reproduce mouse behavior better than the disRNN does?** If
so, the interpretability trade-off looks considerably worse — we would be paying a behavioral cost
for a model that is neither the best predictor nor the best mimic.

---

## This uses Po-Chen's engine, unchanged — that is the whole point

`aind-disrnn-wrapper/code/post_training_analysis/baseline_rl_analysis.py` (**pckuo**, merged
2026-06-18) already implements the baseline-RL generative rollout. **It had never been run**: no
baseline run in either W&B project carries a single `switch_triggered` / `history_dependent` key,
and study 01 kept its RL report (r8, likelihood) and its generative report (r9, GRU-only) apart.

We call it as-is. That is what makes the comparison legitimate: its `_simulate_alias_sessions` calls
**`build_curriculum_matched_task()`** — the *identical* task construction the disRNN rollouts used,
including the off-curriculum / Random Walk fix from wrapper
[#60](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/60). Same tasks, same trial
counts, same seeded rollouts, same statistics. Nothing is re-implemented here, so nothing can
silently diverge from the disRNN numbers it will sit next to.

## ⚠️ The judgment call: WHICH RL baseline

Po-Chen's engine expects the **per-session** fitting schema (`nwb_name`, `agent_alias`, `params` —
the lab's canonical pipeline, where every session is fit independently).

**But r1's baselines are per-SUBJECT fits**, from our own `baseline_rl` training runs — that is where
`compare-to-threshold 0.7170 / bari 0.7149 / hattori 0.7127` actually come from
(`rl-baseline-{ctt,bari,hattori}@20260713-0102*`).

These are **different models**. A per-session fit is a much stronger, more in-sample baseline. Feeding
the engine its intended input would have put one RL baseline on r4's axes and a different one on
r1's, with both reports calling it "the RL baseline" — so we feed it **r1's per-subject fits**,
expanded across each mouse's sessions. That expansion is faithful, not a fudge: a per-subject fit
*has* constant parameters across that mouse's sessions by construction.

It is also the right analogue of what the disRNN does — one model per mouse, evaluated on that
mouse's behavior — whereas per-session fits would be comparing against a model with far more freedom.

## Two adapters were needed, both mechanical

1. **`hydrate_model_dir()` refuses baseline-RL runs.** It validates `model_type=baseline_rl` as
   supported and *then* unconditionally requires `checkpoints/step_*` — which a run whose "model" is
   a table of fitted parameters has no reason to have. Its baseline-RL support is therefore dead
   code (a real bug in wrapper #59). We assemble the `model_dir` directly; `resolve_model_run()`
   itself handles `baseline_rl` correctly.
2. **Schema expansion** (above): per-subject params → one row per session. `log_likelihood` / `LPT`
   are required by the schema but *unused by the rollout* (which reads only `params`, plus
   `n_trials` / `curriculum_name` / `ses_idx` from the **animal** session join). We pass the
   subject-level values through rather than invent per-session ones, and never report them.

## The same task-parameter caveat as generative-dscan

Rollouts match the task **family**, not its **parameters** (gym defaults; `current_stage_actual` is
unused). This applies identically to the disRNN, the GRU, and these RL baselines, so the
*comparison* is unaffected — only absolute "how mouse-like" claims are.

---

**How it runs.** CPU-only → HPC SLURM, not Beaker (AGENTS §13).

```bash
sbatch --export=ALIAS=ctt,OUT=<dir> submit.sbatch     # + bari, hattori
```

Each job simulates **18,124 sessions across all 614 mice** (zero dropped — every mouse has a fit).

| baseline | W&B run | wrapper alias | r1 held-out LL |
|---|---|---|---|
| compare-to-threshold | [`lmg1i9yd`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/lmg1i9yd) | `ForagingCompareThreshold` | **0.7170** |
| Bari | [`bg3nzqz9`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/bg3nzqz9) | `QLearning_L1F1_CK1_softmax` | 0.7149 |
| Hattori | [`unhmbrk4`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/unhmbrk4) | `QLearning_L2F1_softmax` | 0.7127 |

**Status.** ⏳ running 3/3 (launched 2026-07-14 11:1x PT).
