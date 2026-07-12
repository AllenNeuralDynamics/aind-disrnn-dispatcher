# gru-stage4b-concentration

**Stage 4b · GRU · Dirichlet-concentration sweep**

**What differs (vs `gru-stage4b`).** `gru-stage4b` fixed the per-subject family
mixture at Dirichlet **concentration 0.5** and found session conditioning added
nothing. That null is **confounded**: at α=0.5 subjects are nearly single-family, so
a session's family is fixed by subject identity — there is no within-subject
per-session variation for a session embedding to decode. This variant **sweeps the
concentration** `data.agent.subject_presets.session_switching.concentration ∈
{0.5, 1, 2, 5}` (α<1 sparse → α>1 near-uniform mixtures) so subjects genuinely mix
families across sessions. Embedding size is **fixed at 16** (the best-identifiability
point from `gru-stage4b`) to isolate the concentration × session-conditioning
interaction. Grid: `concentration{0.5,1,2,5} × session_encoding{none,scalar} ×
seed{42,43}` = **16 tasks**.

**Question / prediction.** Does the **{scalar} − {none}** recovery gap (per-session
family decoding, mix-weight R²) **grow with concentration**? Expected: at α=0.5
scalar≈none (subject identity suffices); as α rises the realized per-session family
carries information independent of subject identity, so scalar should pull ahead. A
flat gap even at α=5 would make the original null *fundamental* (subject embedding
suffices); a rising gap makes session conditioning matter exactly when within-subject
session structure exists.

**Readout.** Post-hoc via `analysis/stage4b_recovery.py` (mix-weight R² + per-session
family decoding accuracy), plotted vs concentration, faceted by `{none, scalar}`. The
sweep only trains + logs `likelihood_relative_to_groundtruth`; recovery is scored after.

**Compute.** Synthetic (no S3) → launched low-preemptible on the free AWS + on-prem
**H200** pools via `launch_beaker_resumable.py`. Not gcp-h100: those hosts have only
~100 GB disk and have hit "no space left on device". Checkpoint every 5k steps →
auto-resumes on eviction.

**W&B.** project `embedding_recovery`, group `gru-stage4b-concentration@20260712-055109`.

**Launch status.** First launch (`@20260712-054705`) misconfigured — runs landed in the
`test` project because the resumable launcher sets only the W&B group, not the project
(fixed by adding `wandb.project=embedding_recovery` to the sweep command; now documented
in the beaker-launch skill). Second launch (`@20260712-055109`, Beaker exp `01KXB63DPW7SMKJ35D215MSGS8`) logged
correctly and ran 15/16, but **exposed a generator bug**: at high concentration the
LossCounting family draws enough sessions that `loss_count_threshold_mean` drift+noise
goes slightly < 0, tripping the model's `ge=0` constraint and killing that cell (conc=5.0).
Fixed in [aind-disrnn-wrapper#52](https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper/pull/52)
(clamp the LossCounting/CTT params to their model bounds).

**Current run (`@20260712-060852`, Beaker exp `01KXB73W3S8RDJTMAS5SAV4K18`) — validated,
running.** Re-launched pinned to `WRAPPER_REF=fix/stage4b-losscounting-param-clamp`
(see this run's `launch_record/`) to validate the fix pre-merge on the exact failing
workload: the previously-crashing conc=5.0/scalar/seed=43 cell now generates all 200
subjects cleanly and reached training. **16/16 tasks running** in `embedding_recovery`,
0 failures. Once wrapper#52 merges, the committed template's `WRAPPER_REF=ai_hub_pck_integration`
carries the fix, so future relaunches need no pin.

See the study README Variants index and `analysis/reports/INDEX.md` for the
cross-variant synthesis.
