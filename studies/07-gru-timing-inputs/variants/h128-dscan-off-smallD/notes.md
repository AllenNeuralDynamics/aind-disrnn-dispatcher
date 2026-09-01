# h128-dscan-off-smallD — timing-OFF paired baseline, small cohorts

Experiment `01M0NDS1JYZ58ZC2EHGK5X0K4S` (9 tasks: subject_ratio 0.016, 0.049, 0.163 x 3 seeds).

## Why this variant exists

`d100-bridge` failed its bridge, so study-01's H128 column cannot be the
baseline (diagnosis: unpinned `snapshot=None` -> different cohort draw; see
`../d100-bridge/notes.md`). Every D therefore needs its own paired OFF arm.

## Scheduling design

Measured capacity at launch (`check_gpu_availability.py --beaker`):
onprem-h200 15 schedulable/16, aws-h200 10/32, aws-l40s 4/4, g6e 0 (1367 queued),
p5en 2. Quota: 8 allocated + 64 unallocated slots, `maxWorkloadPriority: high`.

**Split by TIER, not by D.** This variant takes the **unallocated, high-priority preemptible**
tier (`{priority: high, preemptible: true}`) — 9 of the 64 unallocated slots.
Rationale: big-D cells are the longest (~10 h) and the most expensive to lose to
an eviction restart, so they get the non-preemptible slots. The short small-D
cells go to `h128-dscan-off-smallD` on unallocated slots at high priority +
preemptible, where auto-resume makes eviction cheap.

**Both S3-capable H200 clusters are listed** (`onprem-h200`, `aws-h200`) so the
scheduler places freely. This deliberately fixes a design error in the ON arm,
where the D grid was split by D *across* clusters — correlating D with hardware,
so a cluster-level artifact would alias onto the curve's shape. Placement is now
orthogonal to D. GCP clusters are excluded: they cannot reach the AWS S3 cache.

## Launcher bug found and fixed

The first launch (`01M0NDM2SBYQSKG0G6MQWSSBMZ`, cancelled) came back with
`priority: low` despite the template requesting the protected tier.
`launch_beaker_resumable.py` had an unconditional
`task["context"] = {"priority": "low", "preemptible": True}` that **silently
discarded any `context:` the template set** — so the tier design had no effect
and the launch still looked successful. Fixed to honour an explicit template
context and fall back to the burst default otherwise. Verified in the rendered
spec: `priority: high` / `preemptible: false` here, `preemptible: true` for
smallD.

Note the read-back asymmetry: `b.experiment.spec()` echoes `priority` but
returns `preemptible=None`, so confirm the flag from
`launch_record/experiment_resumable_submitted.yaml`, not the API round-trip.
