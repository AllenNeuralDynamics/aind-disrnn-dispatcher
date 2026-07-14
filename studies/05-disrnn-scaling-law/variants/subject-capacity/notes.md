# Variant: subject-capacity — is per-subject capacity what caps held-out transfer?

**What differs.** D=100 (as `dscan-mult2`'s D=100 cells), mult=2, β=1e-3, lr=1e-3 — everything
identical to `dscan-mult2` except two new axes:
`subject_embedding_size ∈ {4, 16, 64}` × `subject_penalty ∈ {β, β/10, 0}` × `seed ∈ {0,1}` =
**18 tasks**.

## The hypothesis

Two findings from this study's `dscan-mult2` arm, at fixed mult=2 / β=1e-3:

| D | 10 | 30 | 100 | 300 |
|---|---|---|---|---|
| `update←subject` openness Σ(1−σ) | 0.850 | 0.984 | 0.780 | **0.640** |
| `choice←subject` openness | 0.002 | 0.001 | 0.001 | **0.001** (shut) |
| `update←latent` (interaction) | 0.161 | 0.481 | 0.774 | **0.922** |
| held-out LL | 0.7101 | 0.7147 | 0.7174 | 0.7165 |

As mice accumulate, the model moves capacity **out of** per-subject channels and **into** shared
dynamics: it abandons personalisation and learns a population-average mechanism. At D=300 it is
personalising through **less than one effective open subject dimension**, and the choice net never
uses subject identity at all.

Meanwhile the disRNN sits **~0.010 below the GRU at every D** — a *flat* gap
(−0.0118 / −0.0102 / −0.0088 / −0.0103) — and only **ties** the best per-mouse RL baseline
(compare-to-threshold, 0.7170), while the GRU **beats** that baseline by +0.0098. The GRU's subject
embedding carries **no bottleneck**.

**Hypothesis: the subject bottleneck — not the interaction bottleneck — is the disRNN's transfer
deficit.**

## Why the two axes are coupled

Widening the embedding *alone* would fail: the penalty would simply close the extra dims and we
would learn nothing (a false negative). So the penalty moves with it.

**`subject_penalty=0` is the GRU limit of the subject pathway** — an unbottlenecked subject
embedding with everything else still disRNN. That makes this a *causal* test: if the −0.010 gap
closes at penalty=0, the subject bottleneck **is** the cause. If held-out LL stays at ~0.717 even
with 64 unpenalised dims, the capacity hypothesis is dead and the deficit lives elsewhere.

**Requires dispatcher [#62](https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher/pull/62)**
(merged `98343d9`), which couples `update_net_subject_penalty` and `choice_net_subject_penalty` to
`subject_penalty`. Before that, those two stayed clamped at β and the embedding remained
bottlenecked **downstream** of the knob — this grid would have returned a false negative. The
launch is pinned to that SHA.

## Why D=100, not D=614

The disRNN-vs-GRU gap is **flat in D**, so the deficit is fully present at D=100 — no need to pay
for D=614 to study it. D=100 is also where the comparison data lives (study 03's entire 48-run
grid, the GRU's D=100 cells, this study's own `dscan-mult2` D=100 cells) and is cheaper
(checkpoint eval ~10 min vs ~50 min at D=614). If a wider embedding closes the gap here, confirm
at D=614.

## Built-in control

The `(embed=4, subject_penalty=β)` corner is **config-identical to `dscan-mult2`'s D=100 cells** —
but those ran on the **pre-merge wrapper**, and this launch resolves `WRAPPER_REF` to a SHA that
**includes** the checkpoint perf fix (wrapper [#56](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/56)).
That fix is *claimed* numerically inert (it only changes what gets built for plotting). These 2 runs
**test that claim**: they must reproduce the old D=100 held-out LL (**0.7188 / 0.7162** at seeds
0/1). Not redundancy — the only validation of a merged change that every future cross-SHA
comparison depends on.

## Launch — split into two Beaker experiments (one W&B group)

**Beaker rejects a spec whose resolved JSON payload is too large, with a misleading
`[code=409] a retryable database conflict occurred`.** Retrying never helps. Measured on this study:

| spec | tasks | resolved payload | result |
|---|---|---|---|
| `mult-beta-d614` | 12 | 32,584 B | ✅ |
| `dscan-mult2` | 15 | 40,447 B | ✅ |
| `subject-capacity` | 18 | **54,405 B** | ❌ 409 |

The ceiling is between 40 KB and 54 KB (likely 48 KiB). **Measure the resolved JSON, not the YAML
file** — YAML aliasing collapses repeated env blocks in the file and understates the payload by
~30%. Fix: split the grid into two 9-task experiments (~23 KB each) that keep the **same**
`WANDB_RUN_GROUP` / `WANDB_RUN_ID` / pinned SHAs, so it stays one logical launch in W&B.

- W&B group: `subject-capacity@20260713-225831` (all 18 tasks)
- Beaker: [`01KXFKA0G7E6X1MPSH39YQXMV7`](https://beaker.org/ex/01KXFKA0G7E6X1MPSH39YQXMV7) (tasks 0–8)
  + [`01KXFKA1ZST5F5M46HSW2C4YEG`](https://beaker.org/ex/01KXFKA1ZST5F5M46HSW2C4YEG) (tasks 9–17)
- Specs: `launch_record/experiment_part{1,2}.yaml`

**Status.** ⏳ running 18/18 (launched 2026-07-13 22:58 PT).
