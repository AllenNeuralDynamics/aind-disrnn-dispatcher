# Study: Include-ignore-trials scaling (3-way output)

**Question.** The `data-scaling-law` study showed per-trial **L/R choice** likelihood is
near a *predictability ceiling* — data-scaling is real but saturates by ~100 mice and the
absolute headroom is small. This study tests a **headroom-ier target**: add the *ignored /
disengaged* trial as a third output class (`output_size` 2→3, L/R/ignore). Engagement is
strongly session-structured (motivation, satiation, drift) and may transfer across mice, so
it is a candidate axis where **data-scaling could finally show** a sustained slope that L/R
lacks. (Roadmap issue **#23**; motivation in `../data-scaling-law/FUTURE_DIRECTIONS.md` §6.)

**The only model/data change** vs `data-scaling-law` is `data.ignore_policy: exclude → include`.
The wrapper auto-derives `output_size = 3` from this and validates it
(`gru_trainer.py:_build_*`, `expected_output_size = 2 if ignore_policy=='exclude' else 3`);
no model-code change is needed. Everything else (GRU H, scalar session-conditioning, λ-forward
schedule, lr, batch, held-out cohort, snapshot pin) is held identical to the `v2-sc-active` /
`nxd-grid` variants so the two studies are directly comparable arm-for-arm.

> **Metric caveat (carry into every report).** 3-way NL has a **different chance baseline**
> than 2-way (uniform 1/3 vs 1/2) and is computed over a **different trial support** (include
> keeps the ignore trials that exclude drops). So the 3-way likelihood numbers are **not
> directly comparable** to the `data-scaling-law` L/R numbers by subtraction. Analyze the 3-way
> curve on its own axis, and for a like-for-like comparison score the **conditional L/R
> likelihood on the shared engaged trials** and the **ignore-class likelihood** separately.

## Variants

| variant | what differs | status | W&B group (launch) | Beaker exp |
|---|---|---|---|---|
| [`validation-2way-vs-3way`](variants/validation-2way-vs-3way/notes.md) | quick D≈10 smoke: `ignore_policy` ∈ {exclude, include}, short `n_steps`; proves the 3-way pipeline end-to-end and gives a first look at the ignore effect | 🏃 running | `validation-2way-vs-3way@20260703-092118` | [`01KWMCHPQ803BCBR0AKP4VZWDW`](https://beaker.org/ex/01KWMCHPQ803BCBR0AKP4VZWDW) |
| [`nxd-3way`](variants/nxd-3way/notes.md) | full N×D grid (mirror of `data-scaling-law/nxd-grid`) with `ignore_policy=include` | 🏃 running (48 tasks) | `nxd-3way@20260703-094210` | [`01KWMDT0ETCM6QW6XADAKEQXP8`](https://beaker.org/ex/01KWMDT0ETCM6QW6XADAKEQXP8) |

W&B project: **`mice_ignore_scaling`** (one project per study; one group per launch).

## Design (nxd-3way — mirrors data-scaling-law r7)

- **Grid:** `hidden_size N ∈ {16, 64, 128, 256}` × `subject_ratio ∈ {0.016, 0.049, 0.163, 1.0}`
  (D ≈ {10, 30, 100, 614}) × `seed ∈ {0, 1, 2}` = **48 tasks**.
- **Fixed:** scalar session conditioning; λ-forward (pretrain 30k, warmup 30k→50k ⇒ full SC @50k);
  `n_steps=150000`; `lr=1e-5`; `batch_size=2048`; gated early-stop @70k; length bucketing on;
  per-checkpoint held-out eval off; `snapshot=20260603`; fixed held-out cohort (`heldout_every_n=5`).
- **The one difference from `nxd-grid`:** `data.ignore_policy=include` (⇒ `output_size=3`).
- **y-axis:** held-out-mouse likelihood from the final `auto_heldout_finetune` (fine-tune subject
  embedding only), same protocol as `data-scaling-law`.

## Analysis targets (after runs settle)

1. **3-way held-out scaling vs D** at each N — does the D-slope stay positive past 100 (unlike L/R)?
2. **Decompose the 3-way likelihood:** conditional L/R on engaged trials vs ignore-class likelihood,
   each vs D — locate *which* component (if any) keeps scaling.
3. **N×D fit** `L = E + A·N^-α + B·D^-β` on the 3-way metric; compare α,β,E to r7's L/R fit.
4. Cross-study overlay: place the 3-way curves beside `data-scaling-law` r1/r7 with the chance-baseline
   caveat annotated.

## Provenance
Branch: `feat/ignore-trials-scaling`. Wrapper image `han-hou/disrnn-wrapper-pck-integration`.
Follows `docs/study-organization.md` + `docs/posthoc-analysis.md` conventions.
