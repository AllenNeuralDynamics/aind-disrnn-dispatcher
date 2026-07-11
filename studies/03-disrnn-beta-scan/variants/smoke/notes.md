# Variant: smoke

**What differs.** 1-task smoke: proves length-bucketing trims the unroll and the resumable checkpoint path works end-to-end, at a short `n_steps`. Not a scientific run.

**W&B group.** `smoke@20260703-191003` (project `disrnn_updnet_bottleneck_ratio_100mice`).


**Result.** length bucketing verified (~1.86× step speedup: 2015→1083 ms/step vs unbucketed baseline); resumable output artifact committed; held-out finetune produced 18 heldout/* keys.

See `launch_record/results.md` for the settled record.
