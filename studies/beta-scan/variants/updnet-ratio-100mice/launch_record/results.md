# Launch record — updnet-ratio-100mice

- **W&B project:** [`disrnn_updnet_bottleneck_ratio_100mice`](https://wandb.ai/AIND-disRNN/disrnn_updnet_bottleneck_ratio_100mice)
- **W&B group:** `None`
- **Wrapper ref (pinned):** `87d93f8c` (see `../../../environment.lock`)
- **Beaker image:** `han-hou/disrnn-wrapper-pck-integration-20260630`
- **Status:** success (? clean finished run(s))

**Headline.** 43/48 clean. Multiplier monotonically closes the interaction bottleneck (Σ(1−σ) 3.11→0.00 at weak β); held-out transfer flat (~0.008 LL range, set by β not multiplier). 5 lost = 2 NaN divergences (lr=5e-3/seed=0) + OOM churn. Feeds r1 + r2.

**Feeds.** analysis/reports/r1 + r2
