# gru-stage4b

**Stage 4b · GRU**

**What differs.** per-session family switching. Grid: {4,8,16}×{none,scalar}.

**Result.** mix-weight R²0.55@D16; per-sess fam 0.62

**W&B.** project `embedding_recovery`, group `gru-stage4b@<launch_id>`, sweep [`nptb5bam`](https://wandb.ai/AIND-disRNN/embedding_recovery/sweeps/nptb5bam).

See the study README Variants index and `analysis/reports/INDEX.md` (r1 GRU ladder / r2 disRNN) for the cross-variant synthesis.
