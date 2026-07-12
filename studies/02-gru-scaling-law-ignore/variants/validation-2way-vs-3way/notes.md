# validation-2way-vs-3way

**What differs.** Smoke + first-look. Two grid cells on the same D≈10 / seed-0 data, sweeping
only `data.ignore_policy ∈ {exclude, include}` (wrapper → `output_size` 2 vs 3). Short training
(`n_steps=20000`, λ-forward compressed to full SC @10k) so both finish fast. g6e L40S, one GPU bundle.

**Why it exists.**
1. Prove the 3-way (`output_size=3`, L/R/ignore) pipeline trains end-to-end on Beaker without
   error (the include path was validated in code but never launched in this stack).
2. Give a first matched look at what adding the ignore class does to held-out likelihood.

**What we expect.** The `exclude` cell should reproduce the familiar ~0.72 L/R held-out likelihood.
The `include` cell trains a genuinely different 3-class head; its raw likelihood is **not**
comparable to the 2-way number (chance 1/3 vs 1/2, different trial support). Success = both cells
reach exit 0, W&B logs a 3-class output for the include cell, and the held-out fine-tune completes.

**Metric caveat.** Do not subtract the two likelihoods. For a fair read, compare the conditional
L/R likelihood on shared engaged trials (exclude vs include) and inspect the ignore-class
likelihood separately — deferred to the analysis pass once the full `nxd-3way` grid runs.

**Status.** 🚧 pending launch (Beaker g6e). W&B group `validation-2way-vs-3way@<launch_id>`,
project `mice_ignore_scaling`.

**Result.** _TBD — fill after the run settles (W&B group link, per-cell exit status, held-out LL,
whether the 3-class head logged correctly)._
