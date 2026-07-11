# Variant: updnet-ratio-100mice-ruleplot-3dcb9217

**What differs.** No-retrain restore of the sparse best run (`3dcb9217`: mult=2, β=3e-4, lr=1e-3, seed=1) with `plot_choice_rule`/`plot_update_rules` on, to emit the interpretability figures. `auto_heldout_finetune=false` (figures only).

**W&B group.** `updnet-ratio-100mice-ruleplot-3dcb9217@20260710-234714` (project `disrnn_updnet_bottleneck_ratio_100mice`).
**Beaker exp.** `01KX7YWA6ZWYCF474NWQK7J5ZV`.

**Result.** Restored + logged fig/choice_rule + fig/update_rule_0 + fig/update_rule_1 (2 open recurrent latents). Figures viewable on W&B (GCS media blobs blocked in sandbox).

See `launch_record/results.md` for the settled record.
