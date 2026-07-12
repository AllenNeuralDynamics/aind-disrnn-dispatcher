# Study: Include-ignore-trials scaling (3-way output)

*Folder `02-gru-scaling-law-ignore` — formerly `ignore-trials-scaling`. W&B project
`mice_ignore_scaling`; most runs logged before the 2026-07-11 rename carry
`meta.study="ignore-trials-scaling"` (63/66 as of the rename; a few smoke/validation
runs are `meta.study="adhoc"` or unset). Filter by W&B **project**, not `meta.study`.*

**Question.** The `01-gru-scaling-law` study showed per-trial **L/R choice** likelihood is
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

## Verdict (2026-07-11 — grid complete, 48/48)

Full results in [`analysis/reports/`](analysis/reports/INDEX.md) (r1 choice, r2 detection);
figure + curated numbers in [`analysis/scaling.json`](analysis/scaling.json) / `fig_scaling.png`.

1. **The 3-way head is free on choice prediction.** Held-out L/R likelihood on *engaged*
   trials sits on or above the 2-way `data-scaling-law` reference at every (H, D) — adding
   the ignore class costs nothing on the shared choice task.
2. **Choice likelihood still scales with D, capacity-dependently.** Same shape as the 2-way
   study: H=16 flattens early (0.7164→0.7210 over D=10→614, +0.0046) while H≥64 keeps
   climbing (H=256: 0.7196→0.7315, +0.0119). Best cell **H=256/D=614 = 0.7315**, still not
   plateaued — so the ceiling is a *capacity* limit, not a data limit, on this axis too.
3. **Ignore detection is real and scales with D.** Ignore-class PR-AUC rises ~0.61→0.64 (larger
   models, D=10→614), far above the ~0.05–0.10 no-skill base rate — the headroom-ier target
   the study was chasing does show a sustained D-slope.
4. **But recall is capped near 0.47** regardless of scale. More mice sharpen the *ranking* of
   ignore-likelihood (PR-AUC) without moving the model off a conservative operating point —
   a genuine detection ceiling, not a data-scarcity artifact.

> **Correctness note.** 12 of 48 cells were initially ignore-backfilled via a restore path that
> over-trained the model (wrong held-out metrics); they were re-scored exactly (held-out
> eval only, native checkpoint) and reproduce native values to <1e-6. Full audit:
> [`analysis/provenance/backfill_history.md`](analysis/provenance/backfill_history.md).

## Variants

| variant | what differs | status | W&B group (launch) | Beaker exp |
|---|---|---|---|---|
| [`validation-2way-vs-3way`](variants/validation-2way-vs-3way/notes.md) | quick D≈10 smoke: `ignore_policy` ∈ {exclude, include}, short `n_steps`; proves the 3-way pipeline end-to-end | ✅ done ([results](variants/validation-2way-vs-3way/launch_record/results.md)) | `validation-2way-vs-3way@20260703-092118` (runs pruned) | [`01KWMCHPQ803BCBR0AKP4VZWDW`](https://beaker.org/ex/01KWMCHPQ803BCBR0AKP4VZWDW) |
| [`nxd-3way`](variants/nxd-3way/notes.md) | full N×D grid (mirror of `data-scaling-law/nxd-grid`) with `ignore_policy=include` | ✅ done — 48/48 ([results](variants/nxd-3way/launch_record/results.md)) | 13 groups (relaunched; see `analysis/scaling.json` → `_meta.wandb_groups`) | [`01KWMEF9TMM391Q5BC15KZXDVQ`](https://beaker.org/ex/01KWMEF9TMM391Q5BC15KZXDVQ) |

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

## Analysis

Run `make -C studies/ignore-trials-scaling` (needs `WANDB_API_KEY`) to regenerate everything:
`analysis/scaling.py` pulls the live grid → `analysis/scaling.json` + `scaling.csv` +
`fig_scaling.png`, then rewrites the report blocks. Reports: [`analysis/reports/INDEX.md`](analysis/reports/INDEX.md).

- ✅ **Choice (r1):** conditional L/R on engaged trials vs D, overlaid on the 2-way reference.
- ✅ **Detection (r2):** ignore-class PR-AUC + recall vs D.
- ⏭️ **Deferred:** the `L = E + A·N^-α + B·D^-β` fit (compare α,β,E to r7's L/R fit) and a
  formal cross-study overlay — the qualitative comparison is in r1; the parametric fit is
  future work if a manuscript needs it.

## Compute / capacity
The `nxd-3way` grid is 48 tasks — a large job. **Before (re)launching, check schedulable
capacity and route accordingly:** `python code/check_gpu_availability.py` (Beaker + HPC).
It reports GPUs that can actually accept a job (Beaker: free *and* not on a cordoned node;
HPC: `Cfg−Alloc` on non-drained `aind` nodes) — the raw `beaker cluster list` / `sinfo`
counts overstate availability. Route narrow-N (H16/H64/H128) to any L40S/A100/V100 pool with
room; H256 needs ≥141 GB (H200 or A100-80G). H200 is used for memory, not speed. See the
beaker-launch / hpc-launch skills and AGENTS §10.

## Provenance
- **Dispatcher branch:** `feat/ignore-trials-scaling`. **Wrapper branch:**
  `feat/ignore-engagement-metrics` (the 3-way held-out decomposition + ignore-class metrics);
  producer commit pinned in [`environment.lock`](environment.lock) and stamped into every
  `scaling.json` as `_meta.wrapper_git_sha`.
- **Beaker image:** `han-hou/disrnn-wrapper-pck-integration-20260630` (entrypoint refreshes
  `WRAPPER_REF`/`DISPATCHER_REF` at container start — no rebuild for code changes).
- **Layout:** `analysis/` (code + curated JSON/CSV/figure + `reports/` + `provenance/`),
  shared helpers in `../util/` (`_meta.py`, `plot_style.py`); follows
  `docs/study-organization.md` + `docs/posthoc-analysis.md`. Derived outputs (JSON/CSV/PNG)
  are committed so results render in-repo; only caches + `__pycache__` are gitignored.
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md).
