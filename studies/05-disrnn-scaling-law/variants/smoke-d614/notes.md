# Variant: smoke-d614

**What differs.** 1 task, full cohort (`subject_ratio=1.0`, D≈614), schedule compressed ~30×
(penalty warmup 250; SC pretrain 1000 + warmup 500 → full SC at 1500; `n_steps=2000`;
checkpoint every 1000). Everything else identical to [`dscan-mult2`](../dscan-mult2/notes.md).

**Why it exists.** disRNN has never been trained above D=100 (study 03). Before committing the
15-task fan-out (~270 GPU-h), prove at the full cohort that (1) the loader resolves ~614 subjects
and the subject-embedding table builds, (2) it fits one 48 GB L40S bundle with no OOM and no
silent multi-GPU grab, (3) the whole schedule runs end-to-end through `auto_heldout_finetune` and
logs `heldout/eval_likelihood`, (4) it does not NaN — and to measure s/step at D=614 so the
fan-out's wall-clock estimate is real rather than extrapolated from D=100.

**Not science.** The compressed schedule leaves the model far from converged. No number from this
run feeds a report; it is a pipeline test only.

**Pass criteria.**
- Loader logs ~614 resolved subjects (`len(resolved_subject_ids)`; expect ~614, exact count is
  seed-dependent — see study 01 README on the ratio→D fuzziness).
- `BEAKER_ASSIGNED_GPU_COUNT == 1` (bundle sizing correct).
- Run reaches `finished`; `heldout/eval_likelihood` present in the summary; no NaN in the loss.
- Record steady-state s/step → expected wall-clock for 60k steps at D=614.

**Status.** ⏳ launched — see `launch_record/`.
