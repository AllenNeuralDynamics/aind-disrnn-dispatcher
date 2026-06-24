# Beaker / AI Hub launch & scheduling playbook

Operational reference for launching jobs on Beaker (AI Hub). The terse guardrails live in
`AGENTS.md` §10–11; this file holds the detail. **Read this before any non-trivial launch.**

## Cluster allowlist (the hard rule)

- **Submit ONLY to clusters whose name contains `hub`** — the team's pools:
  `octo-hub-onprem-h200`, `octo-hub-aws-h200`, `octo-hub-aws-l40s`, `octo.hub-*`, `aihub-*`.
- **NEVER** submit to non-hub clusters (`aipbd-*`, `octo.ai-*`, `siti-*`, `dev-*`) even when
  they show free capacity — they are not ours. When dodging contention, pick a *different
  hub* cluster (e.g. idle `octo-hub-aws-h200` / `octo-hub-aws-l40s`), not a non-hub one.

### Verified exception — `ai1/octo.ai-aws-g6e` (2026-06-23)

This non-hub cluster has `allowPreemptibleRestrictionExceptions: True`, so **low-priority
preemptible** jobs are admitted past its user-whitelist (a `{priority: low, preemptible:
true}` task scheduled in ~3s). It is **AWS** (reaches S3, unlike gcp) with the **same NVIDIA
L40S bundle as `octo-hub-aws-l40s`** (≈93 GiB + 12 CPU/GPU → size `--memory 90GiB --cpu 12`
for 1 GPU); 4 nodes × 4 GPUs = 16 slots. Use it as **extra preemptible burst capacity for
S3-backed offline jobs only**, and only as `low`/preemptible — do **not** assume guaranteed
slots there. Other `octo.ai-*` / `aipbd-*` / `siti-*` clusters remain off-limits unless
similarly verified (probe `cluster get --format json` for `allowPreemptibleRestrictionExceptions`).

## Preferred Cluster Order

For known-good S3-backed jobs that can run as `low`/preemptible, prefer:

1. `ai1/octo.ai-aws-g6e` — first choice. It has the same L40S bundle as `octo-hub-aws-l40s`,
   has had lots of available slots, reaches S3, and L40S has been faster than H200 for our
   current workloads. Use only as the verified low/preemptible exception above.
2. `ai1/octo-hub-onprem-h200` — second choice; many slots and reaches S3, good fallback for
   jobs that need H200 memory.
3. `ai1/octo-hub-aws-l40s` — third choice; reaches S3 and uses the same L40S class, but has
   been more constrained than g6e.

### Cross-cloud S3 caveat

gcp clusters (e.g. gcp-h100) **cannot reach AWS S3** (`aind-scratch-data.s3.amazonaws.com`
DNS fails cross-cloud) — never route S3-backed data jobs to GCP. AWS clusters (l40s, h200,
g6e) reach S3 fine.

## Hard-won scheduling lessons (verified on onprem-H200 + aws-L40s, 2026-06-22)

- **Priority for preemptible fan-outs: use `low`.** Low-priority preemptible jobs burst onto
  spare idle GPUs *beyond* the workspace's unallocated-slot budget (measured ~14 concurrent on
  H200), whereas `normal`-priority preemptible jobs are **capped at the unallocated budget**
  (8) — free physical slots will sit idle while tasks pend. Use `normal` only for a single job
  you need to resist eviction. `autoResume` is **auto-applied** to preemptible jobs — do not
  set it explicitly (Beaker rejects `preemptible` + `autoResume`).
- **Guaranteed slots:** `{priority: normal, preemptible: false}` draws on the non-preemptible
  (allocated) budget — protected indefinitely, never evicted.
- **GPUs are bundled with host CPU/RAM.** A `memory` or `cpuCount` request exceeding **one**
  GPU's bundle makes Beaker assign **multiple GPUs** to a `gpuCount: 1` job (L40s bundle
  ≈ 93 GiB + 12 CPU/GPU, so `memory: 256GiB` → **3 GPUs**, starving other allocated tasks).
  Size `memory`/`cpuCount` to one bundle on the target cluster; big-memory only where the
  workload needs it. Check `BEAKER_ASSIGNED_GPU_COUNT` / `beaker job get` GPUS column.
- **Budget caps ≠ physical capacity.** Tasks pending while physical slots are free ⇒ a budget
  cap (or GPU over-assignment) is binding, not capacity.
- **Per-task cluster/resource splits** aren't in `launch_beaker_resumable.py` (single cluster,
  uniform resources). Render with `--no-submit`, edit `context`/`constraints.cluster`/
  `resources` per task, then `beaker experiment create`.
- **Validate one unit first only when something is untested** — a new cluster, a new resource
  sizing, or a changed spec. Then check the assigned GPUs/resources on the *first scheduled
  job* before trusting the full fan-out (catches over-assignment in one step). For a routine
  repeat of a known-good launch, just fan out directly; don't gate every launch on a one-unit
  probe.

## Verify mechanisms with data before asserting (worked context)

When explaining *why* infra/scheduling/quota behaves a certain way, **pull the actual data
first** (`beaker experiment/job get --format json`, `cluster get`, the W&B API) and cite the
field. Distinguish observed fact from inference — label "verified: …" vs "likely, unconfirmed:
…". Don't present a plausible hypothesis as a conclusion, and when two variables changed at
once, isolate them before attributing cause. (Born from a turn where several confident-but-wrong
infra explanations had to be retracted.)
