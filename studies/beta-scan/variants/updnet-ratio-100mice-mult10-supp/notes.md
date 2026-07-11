# Variant: updnet-ratio-100mice-mult10-supp

**What differs.** mult=10-only supplement (β{3e-4,1e-3,3e-3}×lr{1e-3,5e-3}×seed{0,1}=12 tasks) launched on free H200/L40S capacity to complete the mult=10 column at the short horizon.

**W&B group.** `updnet-ratio-100mice-mult10-supp@20260706-093606` (project `disrnn_updnet_bottleneck_ratio_100mice`).
**Beaker exp.** `01KWW4K1BVG07223K9SMJAHPP3`.

**Result.** 9 clean; 2 deterministic NaN divergences at β=3e-4/lr=5e-3/seed=0 (steps 7510/9740); 1 OOM (retried on g6e). mult=10 fully closes update←latent at every β. Feeds r1 + r2 (mult=10 column).

See `launch_record/results.md` for the settled record.
