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

The two superseded originals were **renamed in Beaker** to `DEAD-superseded-part7-badnode-dns`
and `DEAD-superseded-part8-badnode-dns`, with descriptions pointing at their replacements. They
are kept, not deleted — Beaker has no archive, and `delete` would destroy the provenance these
notes cite (and the result datasets). They show a permanent red "10 failed" badge in the UI;
that is expected. Their tasks are correctly overridden by the resubmits via latest-attempt dedup.

## 6 held-out metrics lost then recovered — READ THIS BEFORE USING `grid.csv` (2026-07-25)

**Six values in `analysis/grid.csv` were recovered post-hoc, not logged natively.** They are
flagged `heldout_backfilled=True` there and `heldout/eval_likelihood_backfilled=True` in W&B.

What happened: six heavily-preempted tasks completed training **and** their held-out stage
(Beaker exit 0, logs ending `All done, goodbye`, committed `disrnn-output-*` and
`*-heldoutper_subject_likelihood` artifacts) — but W&B had already marked each run `crashed` on
a heartbeat timeout, and the **final summary write was silently dropped**. The runs sat frozen at
a mid-training step with **no `heldout/*` key at all**. Beaker saw exit 0 so autoResume never
retried them, and `scaling_report.py`'s `state == "finished"` filter dropped all six cells.
Five were D=300/301 — the sparsest column in the grid.

Detected via a persistent, *growing* gap between the Beaker-finished count and the W&B-finished
count (52–55 vs 48–50). That gap was first waved off as preemption-labelling noise, which was
wrong; the correct move is to cross-reference each exit-0 task's `WANDB_RUN_ID` against its W&B
state **and** whether the metric key is actually present.

Recovered with **no GPU** by `analysis/backfill_lost_heldout.py` (self-discovering, idempotent):
the per-subject held-out table survives as a committed `run_table` artifact, and the scalar is a
pure function of it. The aggregation was **verified, not assumed** —
`heldout/eval_likelihood` is the **trial-weighted GEOMETRIC** mean,
`exp(Σ nᵢ·ln(likᵢ) / Σ nᵢ)`, reproducing natively-logged scalars to ≤5.3e-08 across 5 runs.
The two plausible alternatives are wrong by ~0.004–0.005 — *the same magnitude as the effects
this study measures* — so the script re-validates the formula at runtime and refuses to write on
drift. Full mechanism: beaker-launch skill, `references/resume-extend-rescore.md` §4 and
`references/scheduling-lessons.md` "exit 0 with a missing metric".

## NaN divergences and the determinism probe (2026-07-27)

Three tasks died with `ValueError: NaN in params during session-regularized training` — genuine
failures, not preemptions (exit 1, `was_preempted=False`):

| task | D | mult | β | seed | lr | died at |
|---|---|---|---|---|---|---|
| -008 | 10 | 5 | 3e-4 | 0 | 1e-3 | step 13031 |
| -011 | 10 | 5 | 1e-3 | 1 | 1e-3 | step 13680 |
| -012 | 10 | 10 | 3e-4 | 0 | 1e-3 | step 9420 |

**The pattern is D=10 × high multiplier (5 or 10)** — β varies, seed varies (BOTH 0 and 1 appear),
lr is the safe 1e-3 throughout, and no mult=1 or mult=2 cell has diverged at any D. All three die
in a narrow 9–14k window, i.e. just after the penalty ramp (`n_warmup_steps=7500`) fully engages.

*Likely, unconfirmed:* the multiplier drives the interaction bottleneck's σ toward closure; at D=10
there are only ten subjects to constrain it, so the penalty term dominates and σ saturates, blowing
up the log/gradient term. Note study 03 saw mult=10 diverge only at **lr=5e-3** (at D=100); here it
happens at the *safe* lr, suggesting **small D lowers the divergence threshold** — new relative to 03.

**Probe launched: [`01KYJQ472RT9638M68WWH6HJRK`](https://beaker.org/ex/01KYJQ472RT9638M68WWH6HJRK)**
(spec `launch_record/experiment_nanretry1.yaml`, record `launch_record/beaker_nanretry1.json`).
Re-runs all three cells with **identical** config — verified programmatically that the only fields
differing from the originals are `name` and `WANDB_RUN_ID`. If they re-diverge at the same step the
divergence is deterministic (a strong, citable claim); if they survive it is stochastic.

**Seeds were deliberately NOT changed.** Re-rolling the seed usually does clear a NaN, but seed 1
diverged too, so this is an unstable corner rather than one unlucky draw — substituting a seed that
survives would report only the seeds that happened not to blow up (survivorship bias) and would hide
the instability. Study 03 faced the same choice and reported its divergences. If a value is wanted
for these cells, add seed 2 as a documented *supplement*, never as a replacement.

**Two deliberate deviations from a plain resubmit:**
1. **Fresh W&B run ids.** Unlike the bad-node case, these runs *had logged history* (9–14k steps).
   The id-reuse rule holds only for runs that logged NOTHING — reusing an id on a from-scratch
   restart corrupts history, since Beaker restarts at step 0 while W&B's counter does not rewind.
   The `WANDB_RUN_GROUP` is unchanged, so the analysis still picks these up for the same cell.
2. **Distinct task names** (`nanretry1-0NN`). A new experiment brings a new `/results` dataset, so
   there is no checkpoint and the rerun starts from scratch — what the determinism test wants. The
   distinct names also keep the original failures visible in the status counter instead of being
   silently superseded as "latest attempt" (so the Beaker-side task total is now 83, not 80).

### Probe result (2026-07-27, first data): divergence is STOCHASTIC, not deterministic

| probe | outcome | step | original died at |
|---|---|---|---|
| `nanretry1-008` | **re-diverged** (same `NaN in params` error) | **7500** | 13031 |
| `nanretry1-011` | still training, healthy | 17940 | 13680 — **passed it** |
| `nanretry1-012` | still training, healthy | 21170 | 9420 — **passed it** |

**Two of three sailed past the step where they previously died**, and the one that did re-diverge
did so at a *different* step (7500 vs 13031, ~5500 steps earlier). So with config AND seed held
byte-identical, the failure does not reproduce — neither in occurrence nor in timing.

**The trajectory is not bit-reproducible even at a fixed seed.** That is expected for JAX on GPU
(non-deterministic reductions / XLA autotuning), but it has a concrete consequence here: **`seed`
does not pin the run**, so a NaN at this corner is a *rate*, not a property of a particular seed.
This also reframes the earlier survivorship-bias worry — re-rolling the seed was never the relevant
lever, because simply *rerunning* can yield a survivor.

**Report it as a divergence rate, not a deterministic failure.** Current evidence at D=10 × mult≥5:
4 divergences observed across the original 3 cells plus 1 probe re-divergence, against 2 probe runs
that cleared their original death points. Any cell value obtained from a surviving rerun should be
labelled as such.

**Mechanistic clue, strengthened.** `nanretry1-008` died at step **7500** — exactly
`model.training.n_warmup_steps=7500`, i.e. the step at which the penalty ramp reaches full strength.
Combined with the originals all dying in the 9–14k window (just after that ramp completes), this
supports the hypothesis that divergence is triggered when the full-strength penalty drives the
interaction bottleneck's σ to saturation. *Still inference, not verified* — confirming it would need
the σ/penalty traces around the divergence step.

## Second bad-node incident: 3 tasks wedged on `aidc-h200-prd2` (2026-07-31)

Tasks **-060, -062, -079** sat "pending" for five days. This was reported for several days as
*capacity starvation* — that diagnosis was **wrong**. Beaker was placing them; they were failing
to start, repeatedly, on **one node**: `01KPVKJYXNWNJCH7ZFK0TBXPW5` (hostname `aidc-h200-prd2`,
an onprem H200):

```
Failed to start job: failed to create container: cannot create container:
Error response from daemon: No such image:
gcr.io/ai2-beaker-core/public/d9a3b3uvabos73b7u3j0:latest: image not found
```

**Why this is service-side, verified:** `gcr.io/ai2-beaker-core/...` is *Beaker's* internal
registry namespace, not ours. The image id `01KXCF2EASQ8NV463684PZJ0ZP`
(`han-hou/disrnn-wrapper-main-20260712`) still exists, and the same id ran to completion many
times on other nodes — **including twice on this very node days earlier**. So it is a node-local
registry/pull fault, not a spec error. The node is **not cordoned**, so Beaker keeps treating it
as healthy and re-choosing it: `-060` hit it on 07-27 16:01 and again on 07-31 16:55 — four days
apart, same node, same failure. `nanretry1-008`'s retry hit it too, at the same 16:55 sweep.

**This also explains the false "capacity" reading.** Repeated checks showed
`octo-hub-onprem-h200` with 11–14 GPUs free and an empty queue while these tasks sat pending —
because the free capacity was in the same pool as the one broken node they kept being assigned to.
**Lesson: "pending while capacity is free" is not necessarily starvation — read `job.status.message`
on the latest attempt before concluding anything about the scheduler.**

**Rescue: [`01KYXNJYA7KEY47JXWJGEG4M3Y`](https://beaker.org/ex/01KYXNJYA7KEY47JXWJGEG4M3Y)**
(spec `launch_record/experiment_rescue1.yaml`, record `launch_record/beaker_rescue1.json`).

**Cost, accepted knowingly:** the wedged runs were **65–72 % trained** (steps 76120 / 77100 /
68860 of ~107k) and the rescue restarts them **from scratch**. A new experiment gets a new
`/results` dataset, so there is no checkpoint to resume, and restore/extend cannot help — it
downloads the source run's `training-output` artifact, which is only written at END of training,
so an unfinished run is not extendable. Waiting was the only progress-preserving option, and
Beaker demonstrably keeps re-picking the bad node.

**Deviations (same rationale as the NaN probe):** fresh W&B run ids, because these runs *had*
logged 69–77k steps and the id-reuse rule only holds for runs that logged nothing; and distinct
task names (`rescue1-0NN`) so the wedged originals stay visible rather than being superseded by
latest-attempt dedup.

### Tier-3 thrash: the last 3 cells could not finish on low-preemptible (2026-08-01)

The rescue relaunch (`01KYXNJYA7KEY47JXWJGEG4M3Y`, tier 3 `{low, preemptible: true}`) did not
merely run slowly — it **could not make progress at all**. Measured over ~8 h:

| task | attempts | median survival | max |
|---|---|---|---|
| `rescue1-060` | 31 | **1.6 min** | 9.6 min |
| `rescue1-062` | 38 | **2.0 min** | 27.1 min |
| `rescue1-079` | 34 | **1.4 min** | 3.3 min |

~100 attempts, each evicted after 1–2 minutes. A full run needs ~14 h and checkpoints only
every 10k steps, so **every eviction discarded all progress** — a thrash loop with zero net
advance. Two attempts also died on a *second* bad node, `aws-h200-distinct-cricket`
(`rescue1-079`: `Could not resolve host: github.com`, exit 128; `rescue1-060`:
`_duckdb.IOException` timeout to `aind-scratch-data.s3.amazonaws.com`, exit 1) — an AWS node,
so this is a genuine node network fault, *not* the documented GCP-cannot-reach-S3 caveat.

**Fix: tier 1** — [`01KYYW59S4YT65A9HRPTV7GWEX`](https://beaker.org/ex/01KYYW59S4YT65A9HRPTV7GWEX),
`{priority: normal, preemptible: false}`, drawn from the 4 protected allocated slots, never
evicted. The tier-3 experiment was stopped so the two do not race for the same cells.

**Why not tier 2** (`{normal/high, preemptible: true}`, the 8 unallocated slots): still
preemptible — high priority is evicted *later*, not *never*, which is insufficient at 1.5-min
survival. And per the verified measurement in the beaker-launch skill, normal-preemptible is
**capped at the 8 slots and pends when capped** while `low` bursts *past* the cap, so tier 2
could queue MORE while only partly reducing eviction. The skill's rule applies directly: *a few
must-finish runs → tier 1; tier 2 only once the 4 allocated slots are exhausted.* We need 3.

**Caveat accepted:** `autoResume` is auto-applied only to *preemptible* jobs, so these will not
self-restart if they hit a broken node — resubmit manually if so.

**Verification note:** `experiment.spec()` round-trips the context as `preemptible=None`
(beaker-py omits the default), but the **jobs** report `priority=normal, preemptible=False`.
Check the job level, not the spec echo, when confirming tier.

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
