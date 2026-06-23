# Future directions — validating the foundation-model / data-scaling thesis

**Motivation.** The D-sweep (held-out-mouse likelihood vs # training mice) shows a *real but
weak, fast-saturating* effect: ~85–90% of the +0.005–0.006 gain is captured by D≈100, and the
power-law fit collapsed to Dc≈0 (no sustained slope). Held-out (~0.727) sits just below the
within-subject ceiling (~0.73–0.75, flat H2→H256) — i.e. the model already generalizes to a new
mouse almost as well as it fits training mice, even at small D. So per-trial next-choice
likelihood is **near a predictability ceiling**, and D-alone at fixed N can't move it. That does
**not** validate "big data ⇒ better foundation model"; it tests the wrong axis with a
ceilinged metric. The directions below test axes that *can* show foundation-model scaling.

## 1. Zero-shot / few-shot adaptation metrics (the main one)

Our held-out protocol fine-tunes ONLY the new mouse's subject embedding (rest frozen) for 500
steps, then scores its other sessions. The number we report is the **fully-adapted asymptote** —
which saturates. Foundation-model value usually shows up in **adaptation efficiency**, not the
asymptote. Reframe the held-out eval as an adaptation curve:

- **Zero-shot (K=0):** score the new mouse with NO adaptation — subject embedding at its prior
  (init=zeros, or the population-mean embedding, or nearest-training-mouse prototype). Measures
  how well the *population model alone* predicts a brand-new mouse. We already log step-0 of the
  finetune, so zero-shot is recoverable from existing checkpoints.
- **K-shot:** adapt the embedding on K sessions (or K trials) of the new mouse, score the rest.
  Sweep K = 0, 1, 2, 4, 8, … → an **adaptation curve** (likelihood vs amount of new-mouse data).
- **Summary metrics per model:** (a) zero-shot LL; (b) sessions-to-threshold (#sessions to reach
  e.g. 95% of that model's own asymptote, or a fixed absolute LL); (c) area-under-adaptation-curve.

**The scaling test:** plot zero-shot LL and K-shot efficiency **vs D (and vs N)**. A foundation
model trained on more/diverse mice should (i) have higher zero-shot, (ii) adapt in *fewer shots*,
(iii) maybe reach a higher asymptote. If big-D models adapt to a new mouse in fewer sessions —
**that is the foundation-model signal**, even when the fully-adapted asymptote is flat (which is
what we observe). This is the analog of LLM few-shot scaling: more pretraining → better in-context
/ few-shot, not necessarily lower converged loss.

**Implementation (cheap — reuse checkpoints + the offline finetune harness, wrapper 4f29680):**
add an adaptation-budget sweep to `heldout_offline.yaml` (`heldout_example_sessions_per_subject`
∈ {0,1,2,4,8} and/or finetune steps), log per-K likelihood + the per-step curve (already
checkpointed every 100 steps). No retraining — fine-tune-from-checkpoint only. Extract zero-shot
from existing step-0 evals first (free).

## 2. Joint N×D scaling (IsoFLOP)

Both single-axis sweeps saturate: D-alone (fixed H128) flattens by ~100; N-alone (Po-chen's
train10, fixed D=10) is flat H2→H256. Chinchilla's lesson is that N and D must scale *together* —
you can't see N's value until D is large (more data needs more capacity to exploit) and vice
versa. So the saturation of each single axis is expected and **not** conclusive.

- **Grid:** hidden_size N ∈ {16, 64, 128, 256} × D ∈ {10, 100, 614} (× seeds), held-out LL.
- **Fit** `L(N, D) = E + A/N^α + B/D^β`; report E (irreducible/task-noise floor), α, β with
  bootstrap CIs. Look for the **interaction**: does larger N start to help only at larger D
  (and vice versa)? If yes → joint scaling exists; find the compute-optimal N(D) frontier
  (IsoFLOP: fix compute C ≈ params × trials, vary N vs D, minimize loss). If α,β≈0 and E is hit
  everywhere → **task-noise-limited**, foundation-model scaling absent for this metric (strong,
  publishable null).
- Note two distinct "D"s worth separating: **#subjects** (diversity) vs **#trials/#sessions**
  (volume). `subject_ratio` scales both together. A cleaner decomposition: fix #trials, vary
  #subjects (diversity effect) vs fix #subjects, vary #trials (volume effect).

## 3. Out-of-distribution transfer (the real foundation-model claim)

Held-out *mouse* (same task/rig) is weak transfer. The foundation-model claim is cross-domain:
held-out **task / rig / lab**. Test zero/few-shot transfer to a different task family or a
different rig. Single-task, single-lab, 614-mouse data can't show foundation-model-scale transfer;
this needs broader data (more labs/rigs/tasks, ideally 10³–10⁶ subjects).

## 4. Quantify the current saturation (do first — cheap)

Bootstrap the power-law fit on the existing 15+15 runs: report E (asymptote) and the exponent with
CIs, and "fraction of gain by D=K". Turns "saturates by ~100" into a number with error bars. The
current fit's Dc≈0 already suggests no sustained slope — confirm with CIs.

## Priority
(4) now (free, current data) → (1) zero/few-shot from existing checkpoints (cheap, the most likely
place a foundation-model signal hides) → (2) N×D grid (moderate compute) → (3) OOD transfer (needs
new data). Metric caveat throughout: per-trial choice likelihood is ceilinged; adaptation-efficiency
and OOD transfer have more headroom and are more faithful to the foundation-model thesis.
