# Variant: generative-dscan — 2nd-order (generative-behavioral) validation

**What it is.** Not a training run. For each of the 15 finished `dscan-mult2` cells, roll the trained
disRNN out as a **generative agent** on the task and compare its behavior to the real mouse
(switch-triggered curve; history-dependent switch probability). The disRNN counterpart of study 01's
[r9](../../../01-gru-scaling-law/analysis/reports/r9-generative-behavioral-match.md).

**Why it matters here.** Our first-order metric is **headroom-poor**: held-out likelihood puts the
disRNN only ~0.010 below the GRU and merely level with a per-mouse RL baseline. A generative test can
discriminate where likelihood cannot — *does the interpretable model actually behave like a mouse, or
does it just assign similar per-trial probabilities?* If the generative match collapses while
likelihood barely moves, that is the strongest statement of the interpretability cost. If it holds
up, the disRNN is vindicated on the axis that matters.

**How it runs.** No retraining. Cells are selected by **W&B group + `state=finished`** (`--from-wandb-group`),
each hydrating its checkpoint from the `disrnn-output-<run_id>` artifact
(wrapper [#59](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/59)).
**`--checkpoint-policy final`**, *not* `best_eval`: the checkpoint index records only the last 3
checkpoints, so `best_eval` could score a different model than the one whose held-out numbers this
study published.

---

## ⚠️ Caveat to carry into any conclusion: we match the task FAMILY, not its PARAMETERS

The rollout is "curriculum-matched", and after wrapper
[#60](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/60) each session is matched to
the task family the animal **actually ran** — coupled vs uncoupled, baiting vs without-baiting. That
is a real improvement (before it, off-curriculum sessions were silently simulated as a default
*uncoupled-baiting* task even when the animal ran *Coupled Baiting*; ~17% of sessions in the D=10
cohort).

**But the family is where the matching stops.** The task is then built with the gym's **default
block/reward parameters**. `current_stage_actual` is logged and *unused*, so:

- a curriculum that spans **multiple stages** — each with its own reward probabilities, block
  structure, and sometimes a different task — is collapsed into **one generic task**;
- the simulated environment is therefore *not* the environment that mouse experienced in that
  session, only the same broad family.

The real per-(curriculum, stage) parameters live in
[`aind-foraging-behavior-bonsai-automatic-training`](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training).
This is a **known, deliberate limitation** (see the `IMPORTANT LIMITATION (TODO)` comment on
`_build_curriculum_matched_task`), **not** something introduced here, and it applies equally to study
01's r9.

**What that means for reading the result.** The model-vs-animal match is measured against an animal
behaving in its *real* environment and a model behaving in a *generic* one of the same family. Any
mismatch is therefore an **upper bound on the model's true error** — part of it belongs to the
environment, not the model. Comparisons *across* D (and against the GRU, which has the same handicap)
stay valid, because the handicap is identical in every cell. Absolute "how mouse-like is it" claims
are not.

Not worth fixing now (Han, 2026-07-14). Faithfully reproducing per-stage parameters is tracked in
study 01's `FUTURE_DIRECTIONS.md`.

---

**Status.** ⏳ running 15/15 — Beaker [`01KXGA97R3JSCWES7KYPADEMJ7`](https://beaker.org/ex/01KXGA97R3JSCWES7KYPADEMJ7),
W&B group `generative-dscan-mult2@<launch_id>` in `disrnn_data_scaling`. Pinned to a wrapper SHA
combining PRs #59 + #60 (both open); the runs are reproducible from that immutable SHA regardless of
how those PRs land.

**History.** Two earlier launches were pulled: the first hit the `curriculum_name=nan` crash
(13/15); the second was cancelled because it was pinned to an incomplete fix that would have left the
40 literal-`'None'` sessions on the silent-default path. See
[the study README](../../README.md) status log.
