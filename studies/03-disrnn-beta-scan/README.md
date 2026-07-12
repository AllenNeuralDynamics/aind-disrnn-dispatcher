# Study: update-net-ratio β-scan (disRNN interaction sparsity at 100 mice)

*Folder `03-disrnn-beta-scan` — formerly `beta-scan`. W&B project
`disrnn_updnet_bottleneck_ratio_100mice`; runs logged before the 2026-07-11 rename
carry `meta.study="beta-scan"` (uniform across the sampled runs, but filter by W&B
**project** rather than `meta.study` to be safe).*

**Question.** When many mice are trained together, the disRNN's **interaction
bottleneck** (the update-net's latent×latent gate) does not sparsify as much as
we want — it stays open even after regularization, hurting interpretability.
One hypothesis (Kevin, via Po-Chen) is that the fix is a *separate, stronger*
penalty on that specific gate relative to the base β that regularizes every other
bottleneck. The dispatcher exposes this as `update_net_latent_penalty_multiplier`
(effective multiplier = `update_net_latent_penalty / beta`). This study asks, at
a fixed 100-mouse cohort: **does raising the multiplier actually sparsify the
interaction bottleneck, at what collateral cost to the other bottlenecks, and does
any of it cost held-out-mouse transfer?**

**The one real knob** is the multiplier. Everything else is held identical to the
`data-scaling-law` D=100 GRU arm so the held-out numbers are comparable:
2-way output (`ignore_policy=exclude`), **linear choice net** (`choice_net_n_layers=0`,
Kevin's interpretability request), `latent_size=5`, update-net 5×16, scalar session
conditioning, `batch_mode=random`/`batch_size=2048`, length bucketing on,
`snapshot=20260603`, 100 mice (`subject_ratio=0.163`).

> **Metric caveat (carry into every report).** Bottleneck openness is reported as
> **`total_openness` = Σ(1−σ)** (absolute open capacity; ~0 = fully closed), *not*
> `n_eff_open_frac`. The participation-ratio `n_eff_open_frac` is scale-invariant
> and reports a spuriously high value even when a bottleneck is fully shut — on
> this grid it mis-ranked 19/43 runs and manufactured a false U-shape on the
> multiplier axis. See [`analysis/reports/INDEX.md`](analysis/reports/INDEX.md)
> and [`analysis/provenance/metric_caveat.md`](analysis/provenance/metric_caveat.md).

## Verdict (2026-07-11 — grid complete, 43/48 clean + 9-run mult=10 supplement)

Full results in [`analysis/reports/`](analysis/reports/INDEX.md) (r1 sparsity, r2
transfer); curated numbers in [`analysis/beta_scan_summary.json`](analysis/beta_scan_summary.json)
and the two figures.

1. **The multiplier works: it monotonically closes its target.** Interaction
   (update←latent) openness Σ(1−σ) falls **3.11 → 1.16 → 0.11 → 0.00** across
   mult=1→2→5→10 at weak β=3e-4 (and 1.60→0.81→0→0 at β=1e-3). Monotone, not
   U-shaped — the earlier U-shape was a `n_eff_open_frac` artifact (see caveat).
2. **Strong β=3e-3 is already fully closed at every multiplier**, so the multiplier
   only has leverage at weak/moderate β.
3. **The model compensates — capacity is conserved, not removed.** As the multiplier
   squeezes update←latent, `update←subject` and `choice←latent` *open* (information
   reroutes), the recurrent `latent` closes as collateral, `update←obs`
   (prev choice+reward) stays the most open, and `choice←subject` is shut throughout.
4. **Held-out transfer is flat across the multiplier** (~0.008 LL full range across
   all 12 cells). Sparsifying the interaction bottleneck is essentially free; what
   little variation exists tracks base β (weak/moderate ≈0.716, strong β=3e-3 ≈0.709).
5. **Recommended operating point: mult=2 at weak/moderate β.** It compresses the
   interaction ~2.7× while keeping ~1 functional open channel to interpret, at no
   held-out cost. mult=5/10 collapse the gate to *zero* open channels (nothing left
   to read) and leak the representation into the compensating gates.

> **Stability note.** The `mult=10` corner is NaN-prone at `lr=5e-3, seed=0`
> (2 deterministic divergences at β=3e-4). Those cells carry no usable metrics; the
> (mult=10, β) cells are still covered by 3 clean runs each from other seeds/lr.
> See [`analysis/provenance/launched.json`](analysis/provenance/launched.json).

## Variants

| variant | what differs | status | W&B group (launch) | Beaker exp |
|---|---|---|---|---|
| [`smoke`](variants/smoke/notes.md) | 1-task length-bucketing + resumability proof, short `n_steps` | ✅ done | `smoke@20260703-191003` | — |
| [`updnet-ratio-100mice`](variants/updnet-ratio-100mice/notes.md) | the grid: mult{1,2,5,10} × β{3e-4,1e-3,3e-3} × lr{1e-3,5e-3} × seed{0,1} = 48 | ✅ done — 43/48 clean ([results](variants/updnet-ratio-100mice/launch_record/results.md)) | `updnet-ratio-100mice@20260703-200122` | [`01KWNH6J6YV382HH35GSDWNJAE`](https://beaker.org/ex/01KWNH6J6YV382HH35GSDWNJAE) |
| [`updnet-ratio-100mice-mult10-supp`](variants/updnet-ratio-100mice-mult10-supp/notes.md) | mult=10 supplement on free H200/L40S capacity (9 clean, 2 NaN, 1 OOM) | ✅ done ([results](variants/updnet-ratio-100mice-mult10-supp/launch_record/results.md)) | `updnet-ratio-100mice-mult10-supp@20260706-093606` | [`01KWW4K1BVG07223K9SMJAHPP3`](https://beaker.org/ex/01KWW4K1BVG07223K9SMJAHPP3) |
| [`updnet-ratio-100mice-mult10-oomretry`](variants/updnet-ratio-100mice-mult10-oomretry/notes.md) | OOM-retry of one mult=10 cell (h200→g6e) | ✅ done | `updnet-ratio-100mice-mult10-oomretry@20260706-172047` | [`01KWWZ5WXRD8B0AXYC81T1D7SB`](https://beaker.org/ex/01KWWZ5WXRD8B0AXYC81T1D7SB) |
| [`updnet-ratio-100mice-ruleplot-3dcb9217`](variants/updnet-ratio-100mice-ruleplot-3dcb9217/notes.md) | no-retrain restore of the sparse best run to emit choice/update-rule figures | ✅ done | `updnet-ratio-100mice-ruleplot-3dcb9217@20260710-234714` | [`01KX7YWA6ZWYCF474NWQK7J5ZV`](https://beaker.org/ex/01KX7YWA6ZWYCF474NWQK7J5ZV) |
| [`updnet-ratio-100mice-ruleplot-45646c46`](variants/updnet-ratio-100mice-ruleplot-45646c46/notes.md) | same for the least-sparse (by openness) run | ✅ done | `updnet-ratio-100mice-ruleplot-45646c46@20260710-234908` | [`01KX7YZSF25GNZJWFM0SJEBEV3`](https://beaker.org/ex/01KX7YZSF25GNZJWFM0SJEBEV3) |

W&B project: **`disrnn_updnet_bottleneck_ratio_100mice`** (one project per study; one group per launch).

## Design (updnet-ratio-100mice)

- **Grid:** `update_net_latent_penalty_multiplier ∈ {1,2,5,10}` × base `β ∈ {3e-4,1e-3,3e-3}`
  × `lr ∈ {1e-3,5e-3}` × `seed ∈ {0,1}` = **48 tasks**. Effective multiplier is
  recovered post-hoc as `round(update_net_latent_penalty / beta)` (the dispatcher
  consumes and drops the multiplier field before training).
- **Fixed:** 2-way (`ignore_policy=exclude`); linear choice net (`choice_net_n_layers=0`);
  `latent_size=5`; update-net 5 layers × 16 units; scalar session conditioning
  (pretrain 30k, warmup 20k); `n_warmup_steps=7500` (disRNN penalty ramp — NOT
  early-stopping; disRNN has none); `n_steps=60000` (staged short horizon,
  resumable/extendable); `batch_mode=random`, `batch_size=2048`; length bucketing on;
  `snapshot=20260603`; 100 mice (`subject_ratio=0.163`).
- **y-axis:** held-out-mouse likelihood from the final `auto_heldout_finetune`
  (fine-tune subject embedding only), same protocol as `data-scaling-law`.

## Analysis

Run `make -C studies/beta-scan` (needs `WANDB_API_KEY` only if re-pulling the grid;
the committed `analysis/beta_scan_final_grid.csv` is the source of truth). The single
producer `analysis/beta_scan_report.py` writes `beta_scan_summary.{json,csv}` + the
two figures, then rewrites the report blocks. Reports:
[`analysis/reports/INDEX.md`](analysis/reports/INDEX.md).

- ✅ **Sparsity (r1):** six-family openness vs multiplier, by base β.
- ✅ **Transfer (r2):** interaction openness + held-out likelihood vs multiplier.
- Legacy exploratory script `analysis/beta_scan_analysis.py` (old σ<0.1 `frac_open`
  metric, pre-correction) is retained but **superseded** by `beta_scan_report.py`.
- Backfill producers (`analysis/backfill_sparsity_metrics.py`,
  `backfill_sparsity_metrics_stepwise.py`) document how the threshold-free metrics
  were computed offline from checkpoints and pushed to W&B.

## Provenance
- **Dispatcher branch:** `pr/beta-scan-clean` (→ PR #39, base `ai_hub_pck_integration`).
  **Wrapper:** metrics-producing commit pinned in [`environment.lock`](environment.lock)
  and stamped into `beta_scan_summary.json` as `_meta.wrapper_git_sha`.
- **Beaker image:** `han-hou/disrnn-wrapper-pck-integration-20260630` (entrypoint
  refreshes `WRAPPER_REF`/`DISPATCHER_REF` at container start — no rebuild for code changes).
- **Layout:** `analysis/` (code + curated JSON/CSV/figures + `reports/` + `provenance/`),
  shared helpers in `../util/` (`_meta.py`, `plot_style.py`); follows
  `docs/study-organization.md` + `docs/posthoc-analysis.md`. Derived outputs are
  committed so results render in-repo; only caches + `__pycache__` are gitignored.
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md).
