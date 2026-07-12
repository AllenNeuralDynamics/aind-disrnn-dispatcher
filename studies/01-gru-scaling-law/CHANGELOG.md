# Changelog — `studies/data-scaling-law/`

Per-study log per [`docs/posthoc-analysis.md`](../../docs/posthoc-analysis.md). One
entry per merged PR (or coherent unreleased batch). Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

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
