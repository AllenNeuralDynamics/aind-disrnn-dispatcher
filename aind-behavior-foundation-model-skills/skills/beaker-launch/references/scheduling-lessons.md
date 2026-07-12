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

## Priority tiers (verified on onprem-H200 + aws-L40s, 2026-06-22)

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
