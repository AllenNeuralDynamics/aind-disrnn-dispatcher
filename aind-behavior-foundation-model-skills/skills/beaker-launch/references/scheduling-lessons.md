# Beaker scheduling — verified lessons & mechanisms

Deep detail behind the hard rules in SKILL.md. (Absorbed from the former
`docs/beaker-playbook.md`; this file is now the canonical home.)

## The verified g6e exception (2026-06-23)

`ai1/octo.ai-aws-g6e` is non-hub but has
`allowPreemptibleRestrictionExceptions: True`, so **low-priority preemptible**
jobs are admitted past its user-whitelist (a `{priority: low, preemptible: true}`
task scheduled in ~3 s). It is **AWS** (reaches S3, unlike GCP) with the **same
NVIDIA L40S bundle as `octo-hub-aws-l40s`** (≈93 GiB + 12 CPU/GPU → size
`--memory 90GiB --cpu 12` for 1 GPU); 4 nodes × 4 GPUs = 16 slots. Use it as
**extra preemptible burst capacity for S3-backed offline jobs only** — never
assume guaranteed slots there. Other `octo.ai-*` / `aipbd-*` / `siti-*` clusters
remain off-limits unless similarly verified (probe
`beaker cluster get --format json` for `allowPreemptibleRestrictionExceptions`).

## The verified p5en exception (2026-07-12)

`ai1/octo.ai-aws-p5en` is the second non-hub `octo.ai-aws-*` cluster verified the
same way: `beaker cluster get` reports `allowPreemptibleRestrictionExceptions:
True` (identical to g6e), so **low-priority preemptible** jobs are admitted past
its user-whitelist. It is **AWS** (reaches S3) with **NVIDIA H200 141 GB** GPUs,
**3 nodes × 8 = 24 slots**. This is the **preemptible route to H200 memory** — use
it for wide `hidden_size=256` (which OOMs a 48 GB L40S) when the on-prem H200 pool
is full, as burst capacity only (never assume guaranteed slots). Bundle sizing is
the H200 shape, not the L40S `--memory 90GiB --cpu 12` — size to one H200 bundle
and confirm `BEAKER_ASSIGNED_GPU_COUNT=1`. As with g6e, H200 is chosen here for
**memory, not speed**.

## Priority tiers (verified on onprem-H200 + aws-L40s, 2026-06-22)

**Our workspace (`ai1/aind-dynamic-foraging-foundation-model`) has three tiers of
job protection** — pick the tier by how much the job must resist eviction:

| Tier | How to submit | Budget | Eviction |
|---|---|---|---|
| **1. Allocated / protected** | `{priority: normal or high, preemptible: false}` | **4 allocated slots** | never evicted (guaranteed) |
| **2. Unallocated / preemptible-normal** | `{priority: normal/high, preemptible: true}` | **8 unallocated slots** (hard cap) | evicted under contention; auto-resumes |
| **3. Low preemptible (burst)** | `{priority: low, preemptible: true}` | **~unlimited** (best-effort spare GPUs, *ignores* the 8-slot cap) | evicted first; auto-resumes |

Rules of thumb: default fan-outs → **tier 3** (`priority: low`) for max throughput;
a few must-finish runs → **tier 1** (4 protected slots); **tier 2** only when a job
must resist eviction but you've exhausted the 4 allocated slots. `maxWorkloadPriority`
for this workspace is `high` (verified 2026-07-12), so `high` is available above
`normal`. The **4 / 8** budget figures are from the 2026-06-22 measurement (the beaker
CLI does not expose the workspace's slot budget directly — re-derive by watching where
`normal`-preemptible starts pending vs. bursting).

- **Low-priority preemptible jobs burst onto spare idle GPUs *beyond* the
  workspace's unallocated-slot budget** (measured ~14 concurrent on H200), whereas
  `normal`-priority preemptible jobs are **capped at the unallocated budget** (8)
  — free physical slots sit idle while tasks pend. Use `normal` only for a single
  job that must resist eviction.
- `autoResume` is **auto-applied** to preemptible jobs — do not set it explicitly
  (the v2 schema rejects `preemptible` + `autoResume`:
  `Error: preemptible cannot be set with min_runtime or auto_resume`). Confirm
  with `beaker experiment spec <id>`.
- **Guaranteed slots:** `{priority: normal, preemptible: false}` draws on the
  non-preemptible (allocated) budget — protected indefinitely, never evicted.
- **Budget caps ≠ physical capacity.** Tasks pending while physical slots are free
  ⇒ a budget cap (or GPU over-assignment) is binding, not capacity.

## GPU bundles & over-assignment

GPUs are bundled with host CPU/RAM. A `memory` or `cpuCount` request exceeding
**one** GPU's bundle makes Beaker assign **multiple GPUs** to a `gpuCount: 1` job
(L40S bundle ≈ 93 GiB + 12 CPU/GPU, so `memory: 256GiB` → **3 GPUs**, starving
other allocated tasks). Size to one bundle on the target cluster; big-memory only
where the workload needs it. Check `BEAKER_ASSIGNED_GPU_COUNT` /
`beaker job get` GPUS column on the first scheduled job.

## Cross-cloud S3

GCP clusters (e.g. `gcp-h100`) **cannot reach AWS S3**
(`aind-scratch-data.s3.amazonaws.com` DNS fails cross-cloud) — never route
S3/DB-backed data jobs to GCP. AWS clusters (l40s, h200, g6e) reach S3 fine. The
trial/session database is public AWS S3
(`s3://aind-scratch-data/aind-dynamic-foraging-cache`, us-west-2).

## Verify mechanisms with data before asserting

When explaining *why* infra/scheduling/quota behaves a certain way, **pull the
actual data first** (`beaker experiment/job get --format json`, `cluster get`,
the W&B API) and cite the field. Label observed fact ("verified: …") vs inference
("likely, unconfirmed: …"); don't present a plausible hypothesis as a conclusion;
when two variables changed at once, isolate them before attributing cause. (Born
from a turn where several confident-but-wrong infra explanations had to be
retracted.)
