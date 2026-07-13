# Changelog — `studies/data-scaling-law/`

Per-study log per [`docs/posthoc-analysis.md`](../../docs/posthoc-analysis.md). One
entry per merged PR (or coherent unreleased batch). Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — classical-RL baseline suite (Bari / Hattori / CTT)

**2026-07-13.** Fit the full classical-RL baseline suite that issue #20 asks for, on the
same fixed 149-mouse held-out cohort as the GRU. Three new variants
(`rl-baseline-{bari,hattori,ctt}`), each run with `heldout_refit.skip_train_fit=false` so
it fits **both** cohorts — 614 training mice *and* 149 held-out mice — in one run.

### Added
- `variants/rl-baseline-{bari,hattori,ctt}/` — sweeps + notes. Each selects a **standalone**
  model config (`model=baseline_rl_<m>`), which is load-bearing for CTT:
  `ForagerCompareThreshold.__init__` accepts only `(choice_kernel, params)` and the trainer
  calls `agent_class_obj(**agent_kwargs, seed=seed)` **unfiltered**, so a deep-merged config
  would hand it the Q-learning kwargs and `TypeError`.
- `analysis/rl_baseline.py` — generalized from one model to the suite; pulls each model's
  held-out per-subject table, computes paired GRU−RL per model, and identifies the
  best-of-breed. Falls back to the run's output dir on `/allen` when the W&B artifact API
  is unavailable (it intermittently is).
- Per-subject fitted parameters for the **614 training mice** are now available for the
  first time — the `model2` artifact the embedding analyses (#24 / #27) require. The old
  `rl-baseline-simple` could not supply this: it ran `skip_train_fit=true` and only ever
  touched held-out mice, so its parameters covered mice that have **no** subject embedding.

### Changed — ⚠️ corrects a published number
- **r8's headline margin was measured against the wrong baseline.** It reported the GRU
  beating classical RL by **+0.0136**, fit against Bari alone. **Bari is not the strongest
  classical model**: compare-to-threshold beats it (pooled 0.71704 vs 0.71491). Against the
  *best* classical model the GRU's D=614 margin is **+0.0113** — the old figure overstated
  the gap by ~20%. r8, the reports INDEX, and the study README are updated.
  - The GRU still beats **all three** models in **every** (variant, D) cell, 100% of mice at
    D≥30. The conclusion is unchanged; the claim is now "beats the best of three", which is
    smaller but far harder to dismiss.
  - **Reproduction check passed:** the new `bari` arm re-fits the same agent as the legacy
    `rl-baseline-simple` (`cdq292n5`) and lands within **+0.00066** — DE noise. It also
    reproduces the published +0.0136 (we get +0.01352), confirming the pipeline.
- Results 1/4/5/7 keep their **Bari** reference band (`band_model` in `rl_baseline.json`),
  since their prose was written against it. Flagged in r8 as a *historical* reference.
  Re-basing those four figures onto CTT is deliberate follow-up, not done here.

### Findings
- **Hattori is the weakest of the three** (0.71267). Asymmetric learning rates (α⁺/α⁻) *hurt*
  held-out generalization relative to a single rate plus a choice kernel — the extra
  parameter buys nothing, and cost 6× CTT's compute to establish.
- CTT is both the **strongest and the cheapest** (4 params, 30 min vs ~3 h).

### Still open (#20)
All three are fit **per-mouse independently** — no D-axis, no population sharing. The GRU's
advantage remains partly a "population vs per-mouse" win by construction. The
hierarchical-Bayes fit (and its VI counterpart, #57) is still the fair comparison, and is
still unbuilt.

## [Unreleased] — posthoc-analysis standard-structure migration

Brings the study folder into conformance with `docs/posthoc-analysis.md`. No
scientific conclusions change.

### Added
- `analysis/_meta.py` helper for the `_meta` provenance block
  (`produced_by`, `produced_at_pt` in PT, `dispatcher_git_sha`,
  `wandb_groups`). Now embedded as the opening key of every committed
  analysis JSON (`nxd_scaling`, `rl_baseline`, `bootstrap_scaling`,
  `report_data`, `mature_fewshot_curve`, `generative_match`).
- `analysis/update_r8.py` — regenerates the paired GRU-vs-RL table in
  `reports/r8-gru-vs-rl-baseline.md` between `<!-- BEGIN result-8 -->` /
  `<!-- END result-8 -->` markers, mirroring `update_final_report_nxd.py`.
- `Makefile` with PHONY `r1..r8` targets and `all`. Targets call producers
  in dependency order (e.g. r7 chains `nxd_scaling.py → rl_baseline.py →
  update_final_report_nxd.py` to handle the cross-producer overlay on
  `fig_nxd_scaling.png`).
- `environment.lock` — `pip freeze` of conda env `disrnn-cpu` (166 packages,
  including the editable `aind-disrnn-wrapper` pin).
- `.gitignore` — `analysis/_cache_*.json`, `analysis/gru_per_subject.json`
  (W&B-pull cache; previously committed at 440 KB, now `git rm --cached`).

### Changed
- `analysis/bootstrap_scaling.py` — replaced the hardcoded
  `studies/data-scaling-law/analysis/...` output path with
  `HERE / "bootstrap_scaling.json"` so the script is cwd-independent
  (Makefile-callable from the study folder).
- `analysis/nxd_scaling.py` — top-level `groups` key replaced by
  `_meta.wandb_groups`; `analysis/update_final_report_nxd.py` migrated
  to read the new location.
- `analysis/rl_baseline.py` — new `list_gru_groups(api)` enumerates the
  matched W&B groups (cheap metadata-only pass) so `_meta.wandb_groups`
  stays spec-conformant (exact group names) even when the GRU per-subject
  pull is served from cache.
- `reports/r8-gru-vs-rl-baseline.md` — frontmatter extends `inputs` with
  `related_scripts: analysis/update_r8.py` and chains it in `reproduce`;
  body adds `<!-- BEGIN/END result-8 -->` markers around the paired table.

### Known gaps (deferred)
- Ad-hoc JSON (`zeroshot_vs_d.json`, `fewshot_curve.json`,
  `mature_sc_verdict.json`, `paired_v1_v2_cell.json`) lack a committed
  producer and so carry no `_meta` block. r4/r5/r6 reports flag this in
  their `inputs.script: ad-hoc` frontmatter.
- Cross-producer figures (`fig_nxd_scaling.png`, `fig_scaling_v1_v2.png`,
  `fig_fewshot_curve.png`, `fig_zeroshot_vs_d.png`) violate the spec's
  "one producer per artifact" rule; `rl_baseline.py` overlays figures
  written by other producers. Workaround: Makefile chains in order so the
  overlay producer runs last.
- `make all` invokes `rl_baseline.py` five times (r1, r4, r5, r7, r8).
  Stamp-file deduplication deferred.

### Provenance
Commits: `a84dd1b` (gitignore) · `c3b299c` (`_meta` nxd) · `342f5ae`
(`_meta` rl) · `52a4000` (`_meta` bootstrap) · `7882357` (`_meta`
report_data) · `acba0c9` (`_meta` mature_fewshot) · `36548d9` (`_meta`
generative_match) · `6b0ecd5` (r8 markers + writer) · `79f1381`
(bootstrap cwd fix) · `56d833b` (Makefile) · `c6889d2` (environment.lock).
