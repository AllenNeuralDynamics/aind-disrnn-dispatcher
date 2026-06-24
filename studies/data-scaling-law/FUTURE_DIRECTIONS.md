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

## On effect sizes (Kevin Miller): small consistent ΔLL is real
Likelihood here is per-trial-normalized (NL = exp(mean_t log p_t)). A *consistent* Δ=+0.001 at
NL≈0.73 ≈ +0.0014 nats/trial → ~0.7 nats over a 500-trial session → ~2× per-session likelihood
ratio, compounding across sessions/mice. So small per-trial deltas (our SC +0.0015 at large D, the
data-scaling residual) are genuine model evidence, not noise (and the per-mouse pairing, p~1e-24,
confirms consistency). Caveat: this is about *evidence/detectability*, not *headroom* — the curve
still saturates because per-trial L/R choice is near a predictability ceiling. Don't dismiss small
consistent effects; do seek headroom-ier metrics (below).

## 5. Generative behavioral-statistics validation (2nd-order; CHEAP, do first)
Beyond next-trial likelihood, roll the model out AS AN AGENT and compare behavioral phenomenology to
the real mouse (win-stay/lose-shift, switch curves, block/ reversal transitions, choice
autocorrelation). Code exists: `run_analysis.py generative` → `post_training_analysis/generative_analysis.py`
(Po-chen). Runs on existing checkpoints (offline, like the held-out re-runs). A model can match LL yet
generate wrong dynamics → orthogonal check; scaling/SC effects invisible in LL may appear here.

> **⚠️ IMPORTANT TODO — generative task params are NOT stage-matched.** The rollout environment
> (`_build_curriculum_matched_task`, Po-chen's code) matches only the curriculum **family**
> (coupled/uncoupled + baiting) and uses the gym's **default** block/reward parameters
> (`rwd_prob_array=[0.1,0.5,0.9]`, `block_min=20, block_max=35`, …). It **ignores
> `current_stage_actual`**, so a curriculum spanning multiple stages — each with its own reward
> probabilities / block structure / sometimes a different task — is collapsed into one generic
> task. The model is therefore rolled out against a *generic* environment and compared to the
> animal behaving under its *actual, stage-specific* tasks — a confound baked into the corr-0.96
> result. **Faithful fix:** instantiate each session's task with its real per-(curriculum, stage)
> parameters, which live in
> [aind-foraging-behavior-bonsai-automatic-training](https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-automatic-training).
> This affects **all** D points, so adopting it means recomputing the existing generative numbers.
> (The 2026-06-23 family-substring fix only made high-D `*2p3*` curricula resolve to the same
> default-param task — it did **not** address this.)

## 6. 3-way output: include ignored trials (more headroom, MODERATE)
Flip `data.ignore_policy=exclude→include` (output_size 2→3: L/R/ignore — wrapper already supports it)
and re-run the D-sweep. Engagement/ignore is strongly session-structured (motivation, satiation, drift)
and may transfer across mice → a learnable axis with headroom that saturated L/R lacks; possibly where
data-scaling finally shows. Note: 3-way NL has a different chance baseline than 2-way → analyze as its
own curve, not directly comparable to the L/R numbers.

## 7. Lick-level modeling: RT + lick counts/timing (HIGHEST headroom, flagship build)
Model the full behavioral output per trial — reaction time, lick number, lick timing pattern — not just
the discrete choice. Continuous/temporal motor output is far from any ceiling, so data should keep
helping (most likely to validate the foundation-model thesis). Big build: new data pipeline (lick
rasters/RT from raw logs), new output heads + likelihoods (point-process/Poisson for lick trains,
continuous/censored model for RT), new eval. Longer-term flagship.

Priority: 5 (cheap, orthogonal) → 6 (config-flip retrain) → 1/2 (few-shot/N×D, in progress) → 7 (flagship).

## 8. Hypothesis: SC's benefit = accounting for curriculum/stage heterogeneity (TEST via mature-only)
This study deliberately includes early stages (`mature_only=false`), hoping session conditioning
absorbs the naive→mature drift. SC's session feature is `session_idx/session_max` (continuous
"position in the mouse's session sequence"), which correlates with training stage — so SC is
well-positioned to do this. **Falsifiable prediction:** if SC's value is stage-accounting, the SC
benefit (v2−v1) should **shrink toward 0 in a mature-only eval** (STAGE_FINAL/GRADUATED only). Fits
what we see: SC's benefit grows with D (shared session-delta learns the across-mouse stage trajectory
better with more data) and shows even at zero-shot (frozen, shared). 
- **Test:** mature-only held-out re-run (configs `heldout_*_mature.yaml`, `mature_only:true`) for v1+v2,
  zero-shot + adapted → compare SC benefit mature-only vs all-stage. Shrinks ⇒ SC was doing the
  stage job (validates the design choice; SC = tool for messy longitudinal data, not a general
  booster). Persists ⇒ SC also captures within-mature drift.
- Caveat: `session_idx/session_max` tracks across-session drift broadly, not stage *specifically*, so
  "persists" wouldn't be surprising; the *shrinkage magnitude* = fraction of SC's value that was stage.
- Complementary (free): the held-out `subject_session_context_state_space` plot — color sessions by
  `current_stage_actual`; if SC encodes stage, the context representation should separate early vs mature.
- Mature-only ALSO addresses the K=1 few-shot crash (adapting on a naive early session); and re-reads
  all our absolute numbers as "all-stage" (mature-only may shift them).

### Mature-only: two levels (A = eval-only now; B = retrain, confirmatory)
- **A (eval-only, CHEAP, doing now):** `heldout_*_mature.yaml` (`mature_only:true`) re-run the held-out
  finetune+test on STAGE_FINAL/GRADUATED sessions only, using the EXISTING all-stage-trained v1/v2
  checkpoints (no retraining). Tests whether SC's held-out benefit is concentrated in the early-stage
  eval sessions (benefit shrinks mature-only ⇒ supports the stage-accounting hypothesis at the eval
  level). Also fixes the K=1 crash + re-baselines numbers as all-stage-vs-mature.
- **B (retrain mature-only, EXPENSIVE, confirmatory — only if A is ambiguous):** train v1 (SC off) and
  v2 (SC on) FROM SCRATCH on mature-only data, then eval. If SC's benefit vanishes when there were no
  early stages to absorb during *training*, that's the definitive verdict on the design choice. ~30
  training runs (a focused v1+v2 × few-D sweep, not the full grid). Decide after A.
