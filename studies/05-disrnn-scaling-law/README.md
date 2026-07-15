# Study: disRNN data-scaling law (the disRNN half of issue #16)

*Folder `05-disrnn-scaling-law`. W&B project
[`disrnn_data_scaling`](https://wandb.ai/AIND-disRNN/disrnn_data_scaling).*

**Question.** Does training the **disRNN** on more mice improve prediction of mice it has never
seen — and does its interaction bottleneck stay interpretable as the cohort grows?

This closes the gap logged on
[issue #16](https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher/issues/16). The issue's
scope is *"Both architectures: GRU and DisRNN"* at the full 600–800-mouse dataset. The GRU half is
done ([`01-gru-scaling-law`](../01-gru-scaling-law), replicated 3-way in
[`02`](../02-gru-scaling-law-ignore)). The disRNN had only ever been trained at a **fixed
100-mouse cohort** ([`03-disrnn-beta-scan`](../03-disrnn-beta-scan)) and on synthetic populations
([`04`](../04-gru-vs-disrnn-embedding-recovery)) — **never on the full dataset**.

## Verdict — all 45 training runs + generative rollouts complete (2026-07-15)

> All three training waves (15+12+18 = 45/45) and `generative-dscan` (15/15) are finished, clean, no
> NaNs. Reports: [r1](analysis/reports/r1-heldout-scaling.md),
> [r2](analysis/reports/r2-sparsity-and-multiplier.md),
> [r3](analysis/reports/r3-subject-capacity.md),
> [r4](analysis/reports/r4-generative-behavioral-match.md).

**1. The disRNN does not saturate like the GRU — it PEAKS at ~100 mice and then DECLINES.**

| D | 10 | 30 | 100 | 300 | 614 |
|---|---|---|---|---|---|
| held-out LL (3 seeds) | 0.7101 | 0.7147 | **0.7174** | 0.7165 | 0.7154 |
| GRU (study 01) | 0.7219 | 0.7250 | 0.7262 | 0.7267 | 0.7268 |
| gap | −0.0118 | −0.0103 | −0.0088 | −0.0102 | −0.0114 |

**Not undertraining** — the confound was tested and ruled out: `checkpoint/eval_likelihood` is flat
over the last checkpoints at both D=99 (Δ −0.0003) and D=614 (Δ +0.0002). D=614 actually fits
*better* within-subject (0.7243 vs 0.7212) while transferring *worse*. More mice genuinely improve
the fit and genuinely hurt transfer **at this operating point**.

**At D=614 the disRNN (0.7154) is BELOW the best per-mouse classical RL baseline** (compare-to-
threshold, 0.7170), which the GRU beats by +0.0098.

**2. The operating point (mult=2, β=1e-3) is WRONG at large D — and that is what causes the
decline.** Wave 2 finds that at D=614 the best cell is **mult=1, β=3e-4 → 0.7211**, which is
+0.0057 above the scaling curve's D=614 point (0.7154), beats the RL baseline, and closes most of
the gap to the GRU. So the "decline with more mice" is *at least partly an artifact of holding a
too-strong penalty fixed as the cohort grows*, not an intrinsic property of the disRNN.

**3. "More mice ⇒ less sparse" holds — but ONLY as a coarse trend.** At fixed mult=2/β=1e-3,
interaction openness rises **0.161 (D=10) → 0.774 (D=100)**, a ~5× opening of the gate. Wave 2
independently shows openness at the *same* (mult, β) is higher at D=614 than study 03's D=100
(mult=2, β=3e-4: **1.16 → 3.09**). The multiplier is still **monotone** at full cohort (β=3e-4:
3.78 → 3.09 → 0.011 → 0.004), so study 03's mechanism claim survives.

> ⚠️ **Do not read fine structure into openness-vs-D above D≈100.** The seed-to-seed SD of the
> interaction openness at D=614 is **0.384** (per-seed: 1.136 / 0.467 / 0.474) — it *swamps* the
> D=300-vs-D=614 difference. Openness is an extremely seed-variable quantity at large cohort. Only
> the coarse D=10 → D≈100 rise is above the noise.

**4. THE HEADLINE — sparsity is no longer free at scale.** Study 03's central selling point was
that held-out transfer is *flat* across the multiplier, so sparsifying the interaction bottleneck
costs nothing. **That breaks at D=614.** At β=3e-4, held-out LL falls
**0.7211 → 0.7181 → 0.7173 → 0.7177** across mult 1/2/5/10 — sparsification costs ~0.004, about
half the disRNN's entire gap to the GRU. At D=100 interpretability was free; at the full cohort
**interpretability and transfer are a genuine trade-off**.

**This one survives its noise check.** Wave 2 is single-seed *by design* (a mechanism check, not an
effect-size estimate), so the claim was tested against the seed-to-seed SD measured at the same
config from wave 1's three D=614 cells: **held-out SD = 0.00046**, pooled over all four runs at that
config (wave 1 seeds 0/1/2 → 0.7157 / 0.7153 / 0.7153, plus wave 2's own seed 42 → 0.7163). The
multiplier effect (0.0039) is **~8.4× that noise**. Held-out is remarkably seed-stable even though
openness is not.

**5. And it does not BEHAVE like a mouse as well as the GRU does — at any D.**
([r4](analysis/reports/r4-generative-behavioral-match.md), 15/15 rollouts.) Rolled out closed-loop
as an agent, the disRNN's behavioral match to the real animal trails the GRU at *every* cohort size:

| D | 10 | 30 | 100 | 300 | 614 |
|---|---|---|---|---|---|
| history-curve corr, disRNN | 0.9379 | 0.9424 | 0.9502 | 0.9631 | 0.9623 |
| history-curve corr, GRU | 0.9619 | 0.9768 | 0.9819 | 0.9838 | 0.9841 |
| gap | −0.024 | −0.035 | −0.032 | −0.021 | −0.022 |

Seed SD on that curve is **0.0008–0.0020** at D ≥ 100, so the gap is **10–20× the noise**. And it
fails in an informative way: **RMSE is comparable to the GRU's** (sometimes better), so the disRNN
gets the average switch *level* right and the *shape* wrong — its curve is flatter than the animal's.
That is what a model looks like when its bottlenecks have pruned history-dependence, and it is
nearly invisible to a per-trial likelihood. **This is the value of the 2nd-order test: it
discriminates where our headroom-poor first-order metric could not.**

> ⚠️ The rollout matches the task **family**, not its **parameters** (gym defaults; `current_stage_actual`
> is unused), so absolute "how mouse-like" claims are an upper bound on the model's true error.
> Cross-D and disRNN-vs-GRU comparisons are unaffected — the handicap is identical in every cell.

**6. The subject bottleneck is NOT the cause of the GRU gap — the mechanism works, the causal
hypothesis doesn't.** ([r3](analysis/reports/r3-subject-capacity.md), 18/18 finished.) `subject_penalty=0`
is the GRU limit of the subject pathway; if the bottleneck were suppressing transfer-relevant
information, removing it should help most. It doesn't, at any of 3 embedding widths:

| embed \ subject_penalty | 1e-3 (unchanged) | 1e-4 | 0 (GRU limit) |
|---|---|---|---|
| 4 | **0.7175** | 0.7132 | 0.7159 |
| 16 | **0.7199** | 0.7166 | 0.7179 |
| 64 | **0.7202** | 0.7182 | 0.7192 |

The lever demonstrably works — subject-channel openness scales with the penalty by up to **~80×**
(update←subject Σ(1−σ): ≈0.7 at `sp=1e-3` → ≈320 at `sp=0`, embed=64) — but held-out likelihood
never rewards the open condition. The *tightest* penalty wins at every width; the fully-open GRU
limit is slightly worse everywhere. The partially-relaxed condition (`sp=1e-4`) is the *worst* of
the three at every width — a non-monotone U-shape that replicates 3 independent times and is not
explained here. Widening the embedding 4→16→64 gives a modest, plateauing gain (best cell:
embed=64/sp=1e-3 → 0.7202) that recovers about a third of the GRU gap (0.0088 → 0.0060) — but width
helps *alongside* the original penalty, not by relaxing it.

> **Regression control.** `embed=4, sp=1e-3` should reproduce `dscan-mult2`'s D=100 cells; it runs on
> a newer wrapper SHA (`c1c4c81`, includes perf fix #56, claimed numerically inert). Both seeds land
> **+0.0007 / +0.0008** above the original values — small, same-sign, doesn't change any conclusion
> here (all comparisons above are same-SHA), but says cross-SHA held-out numbers in this study should
> be read at ±0.001, not treated as exact.

> **Claims made earlier in this study's status log that the data has since falsified.** Kept rather
> than quietly deleted. (a) "The disRNN saturates by ~100 mice like the GRU" — it does not; it peaks
> and declines. (b) "Interaction openness rises *monotonically* with D, so the multiplier must scale
> *up* with D" — **wrong twice.** The monotonicity was an extrapolation from D≤300 made before the
> top of the curve existed; and my first correction ("openness *falls* at D=614, so it is
> non-monotone") was **also wrong** — that drop is inside the seed noise (SD 0.384). The honest
> statement is (3) above: a coarse rise to D≈100, and nothing readable beyond it. Meanwhile the
> *transfer* data argues the penalty should go **down** at large D, not up.

## Design

One axis moves: **D**. Everything else is pinned to study 03's protocol at its recommended
operating point, so this is study 03's model, scanned over cohort size.

- **Swept:** `data.subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}` → D ≈ {10, 30, 100, 300, 614},
  × `seed ∈ {0,1,2}` (seed drives *both* mouse selection and init → seed-averaging captures
  sampling variability; cohorts are **nested** per seed by permutation-prefix sampling).
- **Fixed (study 03's operating point):** `update_net_latent_penalty_multiplier=2`, base `β=1e-3`,
  `lr=1e-3`, `latent_size=5`, update-net 5×16, **linear** choice net (`choice_net_n_layers=0`),
  scalar session conditioning (pretrain 30k + warmup 20k → full SC at 50k), `n_steps=60000`
  (penalty warmup 7500), `batch_mode=random` / `batch_size=2048`, length bucketing on,
  `snapshot=20260603`, `ignore_policy=exclude` (2-way).
- **y-axis:** `heldout/eval_likelihood` from the final `auto_heldout_finetune` (fine-tune the
  subject embedding only on a held-out mouse's sessions, predict its other sessions). The held-out
  cohort is **fixed** (`heldout_every_n=5`) and identical at every D and seed.
- **Second y-axis (disRNN-only):** bottleneck openness **Σ(1−σ)** vs D — see the metric caveat below.

**Comparability.** The D ladder, held-out cohort, snapshot, output target, batch size and metric key
are all identical to [`01-gru-scaling-law`](../01-gru-scaling-law), so the disRNN curve **overlays
the GRU curve** on the same axes and the two power-law fits (`L = E + (Dc/D)^α`) are directly
comparable. Cell-by-cell, the D=100 cells here are comparable to study 03's D=100 grid.

> **Metric caveat (carried from study 03 — apply to every report).** Bottleneck openness is
> **`total_openness` = Σ(1−σ)** (absolute open capacity; ~0 = fully closed), *not*
> `n_eff_open_frac`. The participation-ratio metric is scale-invariant and reports a spuriously
> high value even when a bottleneck is fully shut — on study 03's grid it mis-ranked 19/43 runs and
> manufactured a false U-shape. See
> [`../03-disrnn-beta-scan/analysis/provenance/metric_caveat.md`](../03-disrnn-beta-scan/analysis/provenance/metric_caveat.md).

## Variants

| variant | what differs | status | W&B group (launch) | Beaker exp |
|---|---|---|---|---|
| [`smoke-d614`](variants/smoke-d614/notes.md) | 1 task, full cohort, schedule compressed ~30× — proves the D=614 pipeline before the fan-out | ⏳ running | `smoke-d614@20260713-001936` | [`01KXD5GZ1S112A6M4SJ0A1TK6J`](https://beaker.org/ex/01KXD5GZ1S112A6M4SJ0A1TK6J) |
| [`dscan-mult2`](variants/dscan-mult2/notes.md) | **the scaling curve**: D {10,30,100,300,614} × seed {0,1,2} = 15 tasks at mult=2 | ✅ done 15/15 | `dscan-mult2@20260713-003428` | [`01KXD6CDKKN2CARG16AW4XQRJN`](https://beaker.org/ex/01KXD6CDKKN2CARG16AW4XQRJN) + recovery [`01KXD6PA22ZZW2MJ2CH0JSKSWT`](https://beaker.org/ex/01KXD6PA22ZZW2MJ2CH0JSKSWT) |
| [`mult-beta-d614`](variants/mult-beta-d614/notes.md) | study 03's mult{1,2,5,10} × β{3e-4,1e-3,3e-3} grid re-run at D=614 = 12 tasks | ✅ done 12/12 | `mult-beta-d614@20260713-003501` | [`01KXD6DCD9VGY3G6D3M0JWPB7X`](https://beaker.org/ex/01KXD6DCD9VGY3G6D3M0JWPB7X) + recovery [`01KXD6PBQ8CDVG7RF8S7DJ64MF`](https://beaker.org/ex/01KXD6PBQ8CDVG7RF8S7DJ64MF) |
| [`generative-dscan`](variants/generative-dscan/notes.md) | **2nd-order validation**: roll each of the 15 `dscan-mult2` cells out as a generative agent and compare its behavior to the real mouse (study 01's r9, for the disRNN) | ✅ done 15/15 → [r4](analysis/reports/r4-generative-behavioral-match.md) | `generative-dscan-mult2@20260714-060524` | [`01KXGBPWPEDW5EC9X3V8YSM34C`](https://beaker.org/ex/01KXGBPWPEDW5EC9X3V8YSM34C) |
| [`subject-capacity`](variants/subject-capacity/notes.md) | **is per-subject capacity the transfer cap?** embed{4,16,64} × subject_penalty{β,β/10,0} at D=100 = 18 tasks; penalty=0 is the GRU limit | ✅ done 18/18 → [r3](analysis/reports/r3-subject-capacity.md) | `subject-capacity@20260713-225831` | [`01KXFKA0G7E6X1MPSH39YQXMV7`](https://beaker.org/ex/01KXFKA0G7E6X1MPSH39YQXMV7) + [`01KXFKA1ZST5F5M46HSW2C4YEG`](https://beaker.org/ex/01KXFKA1ZST5F5M46HSW2C4YEG) + embed64 clean recovery [`01KXH3DXG79JTPEZ959TWP02KQ`](https://beaker.org/ex/01KXH3DXG79JTPEZ959TWP02KQ) |

### Bad-node recovery (2026-07-13)

7 of the 27 tasks (1 in `dscan-mult2`, 6 in `mult-beta-d614`) died pre-start on a single on-prem
H200 node, **`aidc-h200-prd2`** (`01KPVKJYXNWNJCH7ZFK0TBXPW5`): `started=None`, and the missing
image was `gcr.io/ai2-beaker-core/...` — **Beaker's own core sidecar image, not ours**. That node's
Docker cannot pull the platform image, so anything scheduled on it dies before the training
container starts. Not a code bug (AGENTS §10, "transient node failure ≠ code bug"), and autoResume
does **not** cover it (it covers preemption, not a failed container start).

Recovery: the 7 task definitions were re-submitted **verbatim** from the saved rendered specs —
same `WANDB_RUN_ID` / `WANDB_RUN_GROUP` (so they stay in the original launch group, no
fragmentation) and the same pinned SHAs — with only `ai1/octo-hub-onprem-h200` dropped from their
cluster list, so they cannot land on the bad node again. Specs:
`variants/*/launch_record/experiment_recovery_submitted.yaml`.

## What each variant answers

1. **`dscan-mult2` → the scaling law.** Held-out transfer vs D, plus bottleneck openness vs D. The
   GRU saturates by ~100 mice (+0.005 LL total from D=10→614); if the disRNN does too, that
   supports study 01's verdict that the *metric* is near a predictability ceiling rather than the
   architecture being the bottleneck. If the disRNN keeps climbing, the saturation was
   GRU-specific.
2. **`mult-beta-d614` → is the D=100 interpretability verdict still valid at scale?** Study 03's
   motivating premise is that the interaction bottleneck fails to sparsify *when many mice are
   trained together* — yet it was only ever tested at D=100. This re-runs its grid at the cohort
   size where the failure was supposed to appear (issue #16, need 3).

**Known risk to watch:** `mult=10` was NaN-prone in study 03 at `lr=5e-3, seed=0`. Both grids pin
`lr=1e-3` to avoid that corner; a divergence at D=614 anyway is a reportable result, not something
to silently retry away.

## Compute

Small disRNN (latent 5, update-net 16×5, linear choice net) → one 48 GB **L40S** bundle
(`ai1/octo.ai-aws-g6e`, 90GiB / 12 CPU / 1 GPU). *Not* H200: the 141 GB requirement in issue #16
is about the wide **GRU** `hidden_size=256`, which this study does not use. Per-step cost is
D-independent (fixed `batch_size=2048`), so the ~18 h/run measured on study 03's D=100 grid carries
to every D — 15 + 12 tasks ≈ **490 GPU-h**, all low-priority preemptible (AGENTS §10 tier 3, bursts
past the 8-slot cap; eviction recovered by autoResume + full-state checkpoints every 10k steps).

## Launch

```bash
conda activate disrnn-cpu
export BEAKER_TOKEN=$(python -c "import yaml;print(yaml.safe_load(open('$HOME/.beaker/config.yml'))['user_token'])")
V=variants/dscan-mult2   # or variants/smoke-d614, variants/mult-beta-d614
python code/launch_beaker_resumable.py \
  --sweep studies/05-disrnn-scaling-law/$V/sweep.yaml \
  --experiment studies/05-disrnn-scaling-law/$V/experiment.yaml \
  --workspace ai1/aind-dynamic-foraging-foundation-model \
  --output-dir studies/05-disrnn-scaling-law/$V/launch_record \
  --label <short-label> --note "<why this run exists>"
```

`--no-submit` renders the spec without launching. The launcher sets the W&B group to
`<variant>@<launch_id>`, injects `DISRNN_META_*` provenance, and resolves `WRAPPER_REF` /
`DISPATCHER_REF` / `FORAGING_MODELS_REF` to immutable SHAs (AGENTS §10).

## Status log

- 2026-07-13 00:19 PT: study scaffolded (three variants); `smoke-d614` launched. It confirmed the
  full-cohort path works: loader selects the whole train pool (618 subjects) and resolves
  **D=614** after the post-fetch drop — *identical* to the GRU arm's D=614 at `ratio=1.0` (checked
  against `mice_data_scaling`: every H128/H64/H16 run at ratio 1.0 also resolves 614). Bundle
  sizing verified: exactly 1 GPU / 12 CPU / 93 GiB per task, no multi-GPU grab.
- 2026-07-13 00:35 PT: both grids fired (27 tasks). 7 died pre-start on the bad node
  `aidc-h200-prd2`; re-submitted verbatim off that cluster (see Bad-node recovery above). All
  27/27 now running.
- 2026-07-13 01:05 PT: **per-step cost confirmed D-independent** — the ~18 h/run budget holds.
  Steady-state s/step, measured from W&B history *deltas* (not `runtime/step` from step 0):

  | D | 10 | 29–30 | 99–101 | 300–301 | 614 |
  |---|---|---|---|---|---|
  | s/step | 0.70–0.82 | 0.79–0.81 | 0.82–0.86 | 0.85–0.93 | 0.99 |

  Flat in D, as the fixed `batch_size=2048` predicts. An earlier apparent 3.4–7.7 s/step at
  D≈300 was an artifact: `runtime/step` from step 0 is swamped by the one-off data-load cost
  (~10–20 min at large D) when a run has only logged 80–230 steps. **Lesson: always take
  steady-state throughput from a delta between two checkpoints, never `runtime/step`.**
  Cohorts also confirmed nested and identical to study 01's GRU arm: D = 10/10/10, 29/30/30,
  99/101/101, 300/300/301, 614/614/614 across seeds 0/1/2.
- 2026-07-13 04:30 PT: 27/27 running, no NaNs — **including all three `mult=10` cells**
  (eval LL 0.7319–0.7320), the corner that diverged in study 03. Within-subject eval likelihood
  is 0.72–0.75 everywhere, tracking study 03's D=100 trajectory. Progress 18–31%.

  **Wall-clock correction: ~18 h/run is too low at large D — expect ~22–24 h at D=614.** Per-step
  *training* cost is D-independent (see above, unchanged), but **checkpoint *evaluation* cost
  scales with D**: the eval scores the full train+eval splits over all resolved subjects, which
  the smoke's log times at **~50 min per checkpoint at D=614** (09:55:07 → 10:48:45 for the step-2000
  checkpoint) versus ~10 min at D=100. With 6 checkpoints (`checkpoint_every_n_steps=10000`) that
  is ~5 h of eval on top of ~16.7 h of training. GPU-hours rise accordingly (~490 → ~560); the
  budget is not threatened and no cell was changed. W&B-derived ETAs already absorb this because
  they come from observed runtime. *If a future disRNN run wants this cheaper, set
  `checkpoint_eval_on_train_split=false` (halves it) or checkpoint less often at large D.*
- 2026-07-13 09:35 PT: 27/27 alive, 44–71% done, **still no NaNs** (all three `mult=10` cells
  healthy: eval LL 0.7215–0.7241). ETAs 3.7 h (D=10) → ~11.5 h (D=614).
  - **`smoke-d614` FINISHED (exit 0)** — the full-cohort pipeline is validated end-to-end at
    D=614: loader → 614-subject embedding → SC schedule → checkpoints → `auto_heldout_finetune`.
  - One `dscan-mult2` job was **preempted** (exit 143 = SIGTERM) and `autoResume` restarted it in
    place; it resumes from its last full-state checkpoint. This is tier-3 preemptible working as
    designed — *not* the bad-node failure mode (which shows `started=None` and needs a manual
    resubmit). Its W&B run briefly shows `crashed`, which is just how W&B records a SIGTERM.
  - **Root cause of the checkpoint cost found — and it is not padding/eval-unroll.** At every
    checkpoint the disRNN trainer called `dl.add_model_results` over the ENTIRE cohort (a
    ~10M-row per-trial frame) plus a full-dataset forward pass with all hidden states — purely to
    plot example sessions for **2 subjects**. `gru_trainer` already avoids this (builds the
    whole-cohort frame only when it is actually persisted, and passes `raw_df` so plotting slices
    per-subject frames on demand); the optimization was never ported to disRNN. Fix + regression
    tests in aind-disrnn-wrapper `fix/disrnn-checkpoint-eval-cost`. It does **not** affect the
    running grid (tasks are pinned to their SHAs), so no restart.
- 2026-07-13 12:39 PT: 27/27 alive, **56–91%** done, still **no NaNs** (all three `mult=10` cells
  fine). No new failures since 09:35 — the two preemptions (exit 143) were both auto-resumed and
  the lagging D=30 cell is catching up. ETAs: ~1–3 h (D≤100), ~5 h (D=300), **~9 h (D=614)** →
  the grid completes around **21:00–22:00 PT tonight**.
  - Wrapper PR [aind-disrnn-wrapper#56](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/56)
    (the checkpoint whole-cohort-frame fix) is **merged**. It does not touch this grid — the 27
    tasks are pinned to pre-merge SHAs — so the ~5 h/run saving lands on the *next* disRNN run.
  - *Early read, not a result (within-subject eval LL, not the held-out y-axis; runs unfinished):*
    at D=614 the eval LL tracks **base β** (β=3e-4 ≈ 0.724, β=1e-3 ≈ 0.722, β=3e-3 ≈ 0.713) and is
    **flat across the multiplier** at fixed β (β=3e-4: 0.7236 / 0.7239 / 0.7257 / 0.7241 for
    mult 1/2/5/10). That is the *shape* study 03 found at D=100 — but the real test is
    `heldout/eval_likelihood` + Σ(1−σ) openness, which only exist once the runs finish. Do not
    quote these numbers.
- 2026-07-13 15:39 PT: 27/27 alive, **no NaNs**, no new failures. **The first cells have finished
  training and entered the held-out fine-tune**: five runs are past `n_steps=60000` (up to step
  67505), which is the `auto_heldout_finetune` phase — the same ~7.5k extra steps study 03's runs
  logged (they ended at 67556). `heldout/eval_likelihood` has not landed for any run yet; it is
  written at the end of that phase, so the y-axis metric appears as each run closes out.
  D=614 cells are at ~42k/60k with ~6.5 h to go → grid completes ~22:00 PT tonight, held-out
  fine-tunes trailing behind.
  - *Note on the ETA column:* runs past 60k show a negative ETA. That is an artifact of
    extrapolating `(60000 − step)`, not a problem — those runs are in the fine-tune phase, which
    the step-based ETA does not model.
- 2026-07-13 22:58 PT: **`subject-capacity` launched (18 tasks).** The D≤301 cells finished and
  produced the study's first real result — the disRNN abandons per-mouse personalisation as the
  cohort grows (`update←subject` openness 0.85 → 0.64 across D=10→300; `choice←subject` shut
  throughout) while the shared interaction gate opens 0.161 → 0.922. It sits ~0.010 below the GRU
  at **every** D and merely ties the best per-mouse RL baseline. So the new variant asks whether
  **per-subject capacity**, not the interaction bottleneck, is the transfer cap. Enabled by
  dispatcher [#62](https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher/pull/62) (merged
  `98343d9`), which couples the three subject penalties so one knob relaxes the whole subject
  pathway — without it the grid would have been a false negative. Details:
  [`variants/subject-capacity/notes.md`](variants/subject-capacity/notes.md).

  **Infra lesson (cost us ~40 min): Beaker rejects an oversized experiment spec with a misleading
  `[code=409] a retryable database conflict occurred`.** It is not retryable and not a conflict —
  retrying forever gets nowhere. Measured: 12 tasks / 32.6 KB ✅, 15 / 40.4 KB ✅, 18 / **54.4 KB
  ❌**. Ceiling is between 40 and 54 KB (likely 48 KiB). **Measure the resolved JSON payload, not
  the YAML file** — YAML aliasing collapses repeated env blocks and understates the true payload by
  ~30% (the file looked like 37 KB). Fix: split into two 9-task experiments sharing one
  `WANDB_RUN_GROUP`. Diagnosis note: a 1-task and a 6-task slice of the *real* spec both submitted
  fine, which is what ruled out "bad spec" and pointed at size — but those slices were **real grid
  tasks**, and cancelling them still left 6 orphaned W&B runs to delete. Probe with **dummy** tasks,
  never with slices of the real grid.

- 2026-07-14 09:42 PT: **`generative-dscan` complete (15/15) → [r4](analysis/reports/r4-generative-behavioral-match.md).
  The disRNN behaves *less* like a mouse than the GRU at every D** — history-curve correlation trails
  by 0.02–0.03, which is **10–20× the seed noise** (SD ≤ 0.002 at D ≥ 100). It fails informatively:
  RMSE is comparable or better, so it gets the average switch *level* right and the *shape* wrong.
  **This is the payoff of the 2nd-order test** — the likelihood gap (~0.010, r1) was real but hard to
  read; the generative gap says *what* the disRNN is getting wrong, and says it well clear of noise.

- 2026-07-14 07:5x PT: **the six `embed=64` cells died a second time — wrapper #58 was a half-fix.**
  `disrnn_trainer` has two near-duplicate plotting blocks; #58 capped the dims and guarded
  `wandb.Image()` in only one of them. The *session-context* plot still enumerated C(64,2)=**2016 raw
  dim pairs per subject** → a **1.69 GP** figure (a strip ~756 ft tall), and the checkpoint block's
  five `wandb.Image()` calls were unguarded — so PIL's `DecompressionBombError` killed the runs at
  checkpoint step 10000, *again*, exactly the failure mode #58 was written to make impossible.
  Fixed in wrapper [#61](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/61)
  (`5cd8b5b`): guard every checkpoint image conversion, **and plot the leading 4 embedding PCs
  instead of raw dim pairs** — C(4,2)=6 panels at *any* width (the same shape the raw grid gave at the
  default embed=4), constant in dim, rotation-invariant, comparable across widths, and far lighter on
  W&B storage. Relaunched: [`01KXGNW6ADG8NSR6R8XWH5NTSH`](https://beaker.org/ex/01KXGNW6ADG8NSR6R8XWH5NTSH).
  **Bottleneck metrics were never at risk** — they come from `params`, not from these figures; the
  cost was ~18 h of wall-clock, not validity.

  **Lesson: guard the conversion, not just the plot.** A cosmetic diagnostic must never be able to
  kill a training run. The dim cap is a nicety; the `try/except` around `wandb.Image()` is the thing
  that makes the class of bug impossible — and #58 shipped the nicety to one code path while leaving
  the actual crash site bare.

- 2026-07-15 08:5x PT: **`subject-capacity` complete (18/18, final numbers) → [r3](analysis/reports/r3-subject-capacity.md).
  The subject bottleneck is NOT the cause of the disRNN's gap to the GRU.** The manipulation worked
  — subject-channel openness scales with the penalty by up to **~80×** — but held-out likelihood
  never rewards the open condition: the tightest penalty (`sp=1e-3`, unchanged) wins at every one of
  3 embedding widths, and the fully-open GRU limit (`sp=0`) is slightly worse everywhere. The
  partially-relaxed condition (`sp=1e-4`) is the *worst* at every width — a non-monotone U-shape that
  replicates 3 independent times and is not explained here. Widening the embedding 4→16→64 gives a
  modest, plateauing gain (best cell: embed=64/sp=1e-3 → **0.7202**) that recovers about a third of
  the GRU gap (0.0088 → 0.0060) — paired with the *original* penalty, not a relaxed one.

  Confirms the earlier note (0-day-prior status): `heldout/eval_likelihood` is written
  **incrementally throughout** `auto_heldout_finetune`, not once at the end — watched one cell climb
  0.668 → 0.708 → 0.714 → 0.715 → 0.717 → 0.7174 across its own finetune checkpoints. From here on,
  only `state == "finished"` values are trusted.

  **Regression control**: `embed=4, sp=1e-3` (newer wrapper SHA, `c1c4c81`, perf fix #56 claimed
  numerically inert) lands **+0.0007 / +0.0008** above `dscan-mult2`'s original D=100 values, same
  sign both seeds. Small, doesn't change any conclusion (all comparisons above are same-SHA), but
  says cross-SHA held-out numbers here should be read at ±0.001.

  **All 45 training runs across the three waves are now finished.** With `generative-dscan` (15/15,
  r4) also done, the study's active-compute phase is complete; `generative-rl-baseline` (RL reference
  lines for r4) remains in flight.
