# mult-d-grid — variant notes

**What.** D×mult×β×seed grid = 80 runs. D∈{10,30,100,300,614} (study 01/03/05 ladder) ×
mult∈{1,2,5,10} (study 03 ladder) × β∈{3e-4,1e-3} × seed∈{0,1}. n_steps=100000 (GRU-parity,
raised from study 05's 60000). Everything else byte-identical to study 03/05 (architecture, SC
schedule, snapshot, batching), so any slice overlays study 05's dscan-mult2 and study 01's GRU curve.

**Why this shape.** Originally scoped as a single-operating-point D-scan fixed at mult=1, β=3e-4
(study 05's wave-2 winner). **r1 overturned that**
([analysis/reports/r1-penalty-selection.md](../../analysis/reports/r1-penalty-selection.md)), a
zero-new-compute selection plot built from existing study 03 (D=100) + study 05 (D=614) runs: it
showed β=3e-4 is simultaneously the *highest held-out* and the *most overfit* penalty (in-sample-vs-
held-out gap grows +0.0027→+0.0083 from D=100→614). β cannot be picked once and scanned around — it
has to be part of the grid. β=3e-3 is dropped: r1 already shows it underfits at both D=100 and D=614
(heldout 0.7107 / 0.7102), so a third sheet would mostly confirm a known-dead result.

**Seed count.** 2, not 3. Study 05 measured held-out SD ≈ 0.0005 at this config family; the effects
this grid resolves (the multiplier's held-out cost, ~0.003–0.006 at D=614 per study 05 finding #4)
are 6–12× that SD, so SEM≈0.0004 at n=2 clears them with room. Openness readouts are seed-noisy
(study 05: SD 0.38 at D=614) — for those, cite study 05's existing 3-seed bars rather than
re-deriving variance from this grid's 2 seeds; top up a 3rd seed later on any borderline cell if
needed (autoResume/extend makes this cheap).

**Provenance.** Launcher writes `launch_record/` here (rendered spec + resolved SHAs + launch_id).
W&B group is `mult-d-grid@<launch_id>`; runs land in project `disrnn_data_scaling` (shared with
study 05 — filter by group). `DISRNN_META_*` provenance injected by the launcher.

**Full-SC window (improved vs 05).** SC fully on at 50k (pretrain 30k + warmup 20k, kept identical
to 03/05). With n_steps=100000 the full-SC window is 50k (was 10k at 05's 60k), matching study 01's
GRU budget — narrows the disRNN-vs-GRU asymmetry study 05 had to caveat.

**Budget-vs-05 caveat.** Because n_steps also moved (60k→100k) alongside the penalty axes, a naive
cell-by-cell diff against 05's dscan-mult2/mult-beta-d614 mixes two changes. `checkpoint/eval_likelihood`
is logged every 10k steps here; read the 60k checkpoint to recover the same-budget comparison point
when isolating the penalty effect (study 05 showed held-out flat 40k–67k at its fixed penalty).

## Launch — split into 8 Beaker experiments (one W&B group)

**Beaker rejects a resolved-JSON payload over ~48 KiB with a misleading `[code=409] a retryable
database conflict occurred`** (retrying the same oversized payload never helps) — the same failure
mode study 05's `subject-capacity` (18 tasks, 54,405 B) hit and documented. `launch_beaker_resumable.py
--sweep sweep.yaml --experiment experiment.yaml` first tried to submit the full 80-task grid as one
experiment (resolved payload 197,546 B ≈ 2,469 B/task — its HTTP request also exceeded the client's
5 s default timeout before any response, masking the underlying size problem). Verified no experiment
was created by that attempt (checked `b.workspace.experiments()`, nothing matching).

**Fix applied in two parts:**
1. `code/beaker_client.py` `get_beaker_client()` now sets `beaker._timeout = 60` (was beaker-py's
   default ~5 s) — matches the existing workaround in `check_gpu_availability.py`. Narrow, isolated
   change; unblocks any future large-payload request through the same client factory.
2. The already-rendered `experiment_resumable_submitted.yaml` (80 tasks, shared W&B group
   `mult-d-grid@20260718-151409` and pinned SHAs, produced by the `--no-submit` dry-render before the
   failed real submit) was split into 8 chunks of 10 tasks each (~24.8 KB/chunk, comfortably under the
   ceiling) and submitted directly via `beaker.experiment.create()`, preserving the identical group and
   SHAs so it reads as one logical launch in W&B despite 8 Beaker experiment IDs. 2 of 8 chunk submits
   hit a transient 409 on first attempt and succeeded on retry; verified no duplicate experiments were
   created (task-count check across all recent experiment IDs).

| part | tasks | payload | Beaker experiment |
|---|---|---|---|
| 1 | 0–9   | 24,798 B | [`01KXVN8CPDJTNFVJD9VCPGF8HH`](https://beaker.org/ex/01KXVN8CPDJTNFVJD9VCPGF8HH) |
| 2 | 10–19 | 24,802 B | [`01KXVN8F5N493BJ4B7KC3FRKD1`](https://beaker.org/ex/01KXVN8F5N493BJ4B7KC3FRKD1) |
| 3 | 20–29 | 24,800 B | [`01KXVN8HQYA8Y6MHNGPDSX900R`](https://beaker.org/ex/01KXVN8HQYA8Y6MHNGPDSX900R) |
| 4 | 30–39 | 24,800 B | [`01KXVN8RZSGET33Z05J7B61NQ0`](https://beaker.org/ex/01KXVN8RZSGET33Z05J7B61NQ0) |
| 5 | 40–49 | 24,802 B | [`01KXVN8VBAEBB07F0T6AKKJMBK`](https://beaker.org/ex/01KXVN8VBAEBB07F0T6AKKJMBK) |
| 6 | 50–59 | 24,798 B | [`01KXVN92ZNTWAQ10YNV36DDGVQ`](https://beaker.org/ex/01KXVN92ZNTWAQ10YNV36DDGVQ) |
| 7 | 60–69 | 24,790 B | [`01KXVN95GHN340BNAWDM91BC3T`](https://beaker.org/ex/01KXVN95GHN340BNAWDM91BC3T) |
| 8 | 70–79 | 24,782 B | [`01KXVN986D229GE1P5CZDHRX8S`](https://beaker.org/ex/01KXVN986D229GE1P5CZDHRX8S) |

- W&B group: `mult-d-grid@20260718-151409` (all 80 tasks), project
  [`disrnn_data_scaling`](https://wandb.ai/AIND-disRNN/disrnn_data_scaling)
- Specs: `launch_record/experiment_part{1..8}.yaml`; full unsplit render + submit record:
  `launch_record/experiment_resumable_submitted.yaml` + `launch_record/beaker_resumable.json`
- Verified at submission time (2026-07-18 ~15:24 PT): 21/80 jobs already `running`, 59 `pending`
  (queued for the low-preemptible burst tier); a sampled running job requested exactly
  `{gpu_count: 1, cpu_count: 12, memory: 90 GiB}` — no multi-GPU over-assignment.

## Recovery — 20 tasks lost to one bad node, resubmitted (2026-07-24)

**Tasks 060–079 (parts 7 and 8, 10 each) all failed within minutes of each other on 2026-07-24
~11:02–11:05 UTC, exit code 128, `was_preempted=False`** — not a preemption, and not the NaN
pattern from the 3 earlier failures. Job logs show the actual cause:

```
[entrypoint] refreshing source from GitHub before the run...
fatal: unable to access 'https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher.git/':
Could not resolve host: github.com
```

All 20 ran on the **same single node** (`01KREKR0ZA15SVAV97CW72WDNZ`) — verified by checking
`job.node` across several of the 20 — a transient DNS/network failure on that node at container
startup, before training (or even `wandb.init()`) began. Per AGENTS §13 "transient node failure ≠
code bug": resubmit, don't debug. Because `was_preempted=False`, autoResume never retried these —
they were genuinely stuck.

**Resubmitted with identical task specs** (same `WANDB_RUN_ID`/`WANDB_RUN_GROUP`/pinned SHAs),
pulled straight from the already-rendered `experiment_resumable_submitted.yaml` — safe because
none of the 20 ever logged anything to W&B (confirmed: `pull_grid.py` still showed exactly 61 runs
seen, unchanged, before and after — the resubmit-eligible W&B run-ID-reuse rule applies cleanly
here since there was zero history to lose). Split into 2×10-task chunks (same payload-ceiling
reasoning as the original launch) and submitted directly via `beaker.experiment.create()`.

| part | tasks | payload | Beaker experiment |
|---|---|---|---|
| 1 | 060–069 | 24,790 B | [`01KYB43A78GH0X95K1W70EFNSZ`](https://beaker.org/ex/01KYB43A78GH0X95K1W70EFNSZ) |
| 2 | 070–079 | 24,782 B | [`01KYB43HX8G6GPSTZFH09MGQZ4`](https://beaker.org/ex/01KYB43HX8G6GPSTZFH09MGQZ4) |

Verified at resubmission (2026-07-24 15:32 PT): 5/20 already scheduled, all on **different** nodes
(none on the bad node); 2/2 chunk submits hit a transient 409 on first attempt and succeeded on
retry (same known pattern), verified no duplicate experiments were created. Specs:
`launch_record/experiment_resubmit20_part{1,2}.yaml`; record:
`launch_record/beaker_resubmit20.json`.

**Launch checklist (for the next large grid).**
1. Verify the current wrapper image (`beaker workspace images ai1/aind-dynamic-foraging-foundation-model`).
2. `python code/check_gpu_availability.py` — route to backend(s) with schedulable GPUs.
3. `git ls-remote origin main | cut -f1` for wrapper/dispatcher/foraging-models → pin SHAs (the
   launcher does this automatically at `--no-submit` render time too).
4. `launch_beaker_resumable.py --sweep sweep.yaml --experiment experiment.yaml --output-dir
   launch_record --label <name> --no-submit` first — check the rendered spec's resolved-JSON size
   (`len(json.dumps(yaml.safe_load(open(...))))`). **If > ~40 KB, split into ≤~15-task chunks and
   submit each directly** (see above) rather than retrying a bare `--no-submit`-less call.
5. Validate GPUs on the first scheduled task (`BEAKER_ASSIGNED_GPU_COUNT=1` / a sampled `job.execution
   .spec.resources`) before trusting the fan-out.
