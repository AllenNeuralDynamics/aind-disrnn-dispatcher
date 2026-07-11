# updnet-ratio-100mice

**What differs.** 1D scan of `update_net_latent_penalty_multiplier ∈ {1,2,5,10}` (Kevin's
knob — harder penalty on the interaction/update-net latent channel), crossed with
`base beta ∈ {3e-4,1e-3,3e-3}` × `lr ∈ {1e-3,5e-3}` × `seed ∈ {0,1}` = **48 tasks**, at
**100 mice** (`subject_ratio=0.163`), 2-way L/R (`ignore_policy=exclude`), with a **linear
choice net** (`choice_net_n_layers=0`).

**Why it exists.** Po-Chen scanned this multiplier only at ≤10 mice; the interaction
bottleneck loses sparsity precisely at *large* cohort, and that regime was never tested. This
is the first update-net-ratio scan at 100 mice. The linear choice net keeps the latent→policy
map interpretable so a sparse interaction net actually buys interpretability.

**How the multiplier works.** `resolve_disrnn_penalties` (wrapper `utils/run_helpers.py`)
computes `resolved[update_net_latent_penalty] = base_beta × multiplier` (base defaults to
`beta` if unset), then removes the multiplier key (idempotent). So `multiplier=1` reproduces
the symmetric baseline; `multiplier=10` penalizes the update-net latent channel 10× harder
than the other bottlenecks. It is the **only** penalty multiplier in the codebase — there is
no subject/embedding multiplier.

**Sparsity readout.** Runs log real-time `bottlenecks/*` scalars (wrapper commit `5cb7154`).
The primary metric is **`final/bottlenecks/update_net_latent_frac_open`** (fraction of
interaction latent bottlenecks with σ < 0.1) — lower = sparser interaction. Also logged:
`total_sigma`, per-family open/closed counts, per-family mean/min σ.

**Compute.** Small disRNN (latent 5, update-net 16×5, linear choice) → routed to **g6e L40S**
(one 90GiB / 12-CPU bundle), NOT `octo-hub-onprem-h200`. Keeping it off H200 also avoids
contending with the running `ignore-trials-scaling/nxd-3way` grid there.

**Staged horizon (this round).** `n_steps=60000` (not 150k) so all four multipliers reach a
comparable, interpretable point fast. `checkpoint_every_n_steps=10000` keeps the full
resumable state uploading per run, so promising cells can be **extended later** to a longer
horizon via `model.training.restore_from_run_id=<source run id>` — it downloads that run's
`disrnn-output-<id>` W&B artifact and continues from its checkpoint (skips warmup) instead of
restarting. Set a larger `n_steps` than the source (loop is `while steps_completed < n_steps`).
Because extending only some cells makes the grid horizon-heterogeneous, read any cross-cell
comparison at a common step (loss-pace logging supports this).

**Length bucketing ON.** `model.training.length_bucketing=true` + `length_bucket_grid=128`
trims each `random`-mode batch's unroll from the global `T_max`≈1488 to the batch's own
session length — **measured ~1.86× throughput** (2015→1083 ms/step, matched config). Requires
`batch_mode=random` (satisfied: batch is `random`/2048, inherited from `mice_snapshot_scaling`).

**disRNN-trainer caveat.** `sweep.yaml` deliberately OMITS `early_stopping` — that exists only
in `gru_trainer`, and the disRNN trainer would error on it. `n_warmup_steps=7500` here is the
disRNN *penalty* warmup, not GRU early-stopping. (`length_bucketing` is now supported by the
disRNN trainer as of wrapper `87d93f8c` — the config declares the keys.)

**Launch (render-first — inspect the rendered spec before any real submit):**
```bash
conda activate disrnn-cpu
WS=ai1/aind-dynamic-foraging-foundation-model

# 1) RENDER ONLY — writes launch_record/experiment_resumable_submitted.yaml, submits nothing
python code/launch_beaker_resumable.py \
  --sweep studies/beta-scan/variants/updnet-ratio-100mice/sweep.yaml \
  --experiment studies/beta-scan/variants/updnet-ratio-100mice/experiment.yaml \
  --workspace "$WS" \
  --label updnet-ratio-100mice \
  --note "update-net latent penalty ratio scan at 100 mice, linear choice net; test whether a harder interaction bottleneck restores sparsity at large cohort (Kevin's idea; gap: ratio only scanned @10 mice before)" \
  --no-submit

# 2) Inspect the rendered grid (48 tasks, per-task overrides, W&B group, L40S sizing), THEN
#    re-run WITHOUT --no-submit to actually launch.
```

**Walltime note (if launched via HPC SLURM instead of Beaker).** A 150k-step disRNN run is
median ~7.6h / p90 ~11.9h at ~3–3.5 steps/s. The `wandb_sweep_gpu.slurm` default
`-t 05:00:00` KILLS full runs mid-training — override with
`--sbatch-extra='--time=16:00:00'`. On Beaker this is handled by preemptible + autoResume.

**Status.** 🚧 scaffolded, `--no-submit` rendered, **not launched** (awaiting go/no-go).
W&B group `updnet-ratio-100mice@<launch_id>`, project `disrnn_updnet_bottleneck_ratio_100mice`.

**Result.** _TBD._
