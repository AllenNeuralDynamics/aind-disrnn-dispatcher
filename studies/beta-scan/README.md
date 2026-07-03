# Study: disRNN bottleneck-β scan (update-net ratio at large cohort)

**Question.** In the disRNN, the *interaction* (update-net) bottleneck does **not stay as
sparse as we want when the number of training mice is large**. A sparse update-net is what
makes the recovered dynamics interpretable (few active interaction channels), so losing
sparsity at scale defeats the purpose of the model. This study asks: **can a harder,
*asymmetric* penalty on the update-net latent channel restore interaction sparsity at 100
mice, without giving up choice-prediction fit?**

**Hypothesis (Kevin, relayed via Po-Chen).** Penalize the **update (interaction) network's
latent input harder than the other bottlenecks**, via the existing
`update_net_latent_penalty_multiplier`, while keeping the **decision (choice) network a pure
linear readout** (`choice_net_n_layers=0`). Rationale:
- The multiplier scales *only* the update-net latent penalty
  (`resolve_disrnn_penalties`: `resolved[update_net_latent_penalty] = beta × multiplier`),
  so raising it squeezes the interaction bottleneck specifically — the exact channel whose
  non-sparsity is the problem — rather than the whole model uniformly.
- A **linear** choice net (no hidden layer) keeps the latent→policy map a single weight per
  latent, so the interpretability gain from a sparse interaction net isn't laundered back
  through a nonlinear readout. (This is Kevin Miller's `disentangled_rnns`; `n_layers=0` ⇒
  empty `choice_mlp_shape` ⇒ linear readout. Validated: **68 of Po-Chen's prior disRNN runs**
  already trained with `choice_net_n_layers=0`.)

**The coverage gap this closes.** Po-Chen *did* scan `update_net_latent_penalty_multiplier`
at **{2, 5, 10}× base — but only at ≤10 mice** (24 diverging 10-mice disRNN runs: 12 in
`mice_multisubject_train10_update_net_latent_penalty_multiplier` at 2/5/10×, + 12 in
`mice_multisubject_train10` at a fixed 5×). At the large cohort we actually care about
(100 mice), **no disRNN β/ratio scan was ever run** — `mice_multisubject_train100` is a GRU,
with no penalty block at all. So the knob most likely to fix large-N sparsity has never been
tested at large N. This study runs it at **100 mice** (`subject_ratio=0.163`, the same cohort
as the `data-scaling-law` / `ignore-trials` D=100 arm).

> **Metric note.** Bottleneck sparsity was previously only visible as the `fig/bottlenecks`
> *image*. This study adds **real-time scalar** sparsity logging to the wrapper
> (`bottlenecks/*` per checkpoint, `final/bottlenecks/*` in `wandb.summary`), including an
> **isolated `update_net_latent`** breakout — mean/min sigma, `n_open`, `n_closed`,
> `frac_open` — which is the direct readout for this hypothesis (the library's aggregate
> `update_bottlenecks_open` mixes subj+obs+latent and hides it).
> Wrapper commit `5cb7154` (branch `feat/bottleneck-sparsity-logging`).

## Variants

| variant | what differs | status | W&B group (launch) | Beaker exp |
|---|---|---|---|---|
| [`updnet-ratio-100mice`](variants/updnet-ratio-100mice/notes.md) | 1D update-net-ratio scan (× base β × lr × seed) at 100 mice, linear choice net | 🚧 scaffolded (not launched) | `updnet-ratio-100mice@<launch_id>` | _TBD_ |

W&B project: **`disrnn_updnet_bottleneck_ratio_100mice`** (one project per study; one group per launch).

## Design (updnet-ratio-100mice)

- **Grid (48 tasks):**
  `update_net_latent_penalty_multiplier ∈ {1, 2, 5, 10}` ×
  `base beta ∈ {3e-4, 1e-3, 3e-3}` ×
  `lr ∈ {1e-3, 5e-3}` ×
  `seed ∈ {0, 1}`.
  - The multiplier is the primary axis (Kevin's knob); base β sets the overall bottleneck
    pressure the multiplier scales from; lr is included because the disRNN lr at 100 mice
    was **never characterized** (Po-Chen's disRNN lr ∈ {1e-4…1e-2} only at ≤10 mice; the
    large-N GRU used lr=1e-5) and lr interacts with how fast the sigmas open/close.
- **Fixed:** 100 mice (`subject_ratio=0.163`); **2-way** L/R (`ignore_policy=exclude`) to stay
  comparable to the `data-scaling-law` baseline; disRNN defaults (`latent_size=5`, update-net
  16×5) **except `choice_net_n_layers=0`** (linear choice net); scalar session conditioning
  (pretrain 30k, warmup 20k); `n_steps=150000`; disRNN penalty warmup `n_warmup_steps=7500`
  (~5% of n_steps); `checkpoint_every_n_steps=10000`; `snapshot=20260603`.
- **disRNN-trainer caveat (baked into `sweep.yaml`):** the disRNN trainer has **no**
  `early_stopping` or `length_bucketing` (those are `gru_trainer`-only). The sweep command
  omits them — do **not** copy the `gru_scaling` sweeps verbatim.

## Readouts (see `analysis/`)

1. **Sparsity vs multiplier** (faceted by lr) — does a harder update-net ratio ⇒ fewer open
   `update_net_latent` bottlenecks at 100 mice? (the core test.)
2. **Test likelihood vs multiplier** — does squeezing the interaction net cost fit?
3. **Tradeoff scatter** (sparsity vs likelihood, colored by multiplier, marker by lr) — find
   the ratio that keeps the interaction sparse without losing choice-prediction accuracy.

## Launch (render-first)

Always render `--no-submit` and inspect `launch_record/experiment_resumable_submitted.yaml`
before a real submit. See the variant `notes.md` for the exact command.

## Provenance

Addresses the HP-sweep item (roadmap **#18**) and the interpretability/mechanism angle
(linear choice net + sparse interaction). Prior scan history and the coverage-gap figure:
project artifacts `pochen_scan_coverage.png`, `pochen_disrnn_beta_scans.csv`.
