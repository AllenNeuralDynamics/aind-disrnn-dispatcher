# Launch record — updnet-ratio-100mice-mult10-supp

- **W&B project:** [`disrnn_updnet_bottleneck_ratio_100mice`](https://wandb.ai/AIND-disRNN/disrnn_updnet_bottleneck_ratio_100mice)
- **W&B group:** `updnet-ratio-100mice-mult10-supp@20260706-093606`
- **Beaker exp:** [`01KWW4K1BVG07223K9SMJAHPP3`](https://beaker.org/ex/01KWW4K1BVG07223K9SMJAHPP3)
- **Wrapper ref (pinned):** `87d93f8c` (see `../../../environment.lock`)
- **Beaker image:** `han-hou/disrnn-wrapper-pck-integration-20260630`
- **Status:** success (9 clean finished run(s))

**Headline.** 9 clean; 2 deterministic NaN divergences at β=3e-4/lr=5e-3/seed=0 (steps 7510/9740); 1 OOM (retried on g6e). mult=10 fully closes update←latent at every β. Feeds r1 + r2 (mult=10 column).

**Feeds.** analysis/reports/r1 + r2
