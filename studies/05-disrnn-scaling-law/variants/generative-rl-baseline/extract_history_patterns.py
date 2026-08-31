#!/usr/bin/env python
"""Freeze the per-pattern history-dependent switch rows out of the cached RL rollout stats.

WHY THIS EXISTS. `reanalyze_stats_only.py` computed the full history-dependent stats for
each baseline and wrote them to
`<dir>/analysis/<wrapper_alias>/history_dependent_switch_stats_no_figures.json` (~1.5 GB
each), but only the trimmed `quantitative_summary` / `delta_significance_summary` blocks
were committed to `rl_rollout_summaries/`. Those carry r / RMSE and per-pattern *deltas*
but NOT the two absolute coordinates (`animal_mean`, `simulated_mean`) that the
model-vs-animal pattern scatter plots, so no scatter could be drawn from the repo. It also
skipped `_save_history_dependent_switch_figures()` wholesale — because the *per-session*
scatters (18,124 points) cost 20-25 min apiece — which took the cheap pattern-comparison
scatter down with them.

WHAT IT DOES. Streams each 1.5 GB file line-by-line and lifts out four top-level blocks:
`config`, `comparison` (pooled rows), `subject_aggregate` (subject-mean rows + panel
summary — what the wrapper's own figure plots), and `session_aggregate`. Top-level keys sit
at exactly two-space indent because the source was written with `json.dumps(indent=2)`, so
a block runs from its own key line to the next line starting with `  "`. That avoids
loading 1.5 GB into memory. Output is ~177 KB per alias, small enough to commit, and is
the frozen source of truth for the figure producer (per AGENTS.md §12 / the
`posthoc-reporting` skill: producers read committed files, never a live handle).

Verification: `subject_aggregate.abstract["3"].summary` must reproduce the already-committed
`quantitative_summary.subject_mean.abstract["3"]` numbers exactly — asserted below.

Usage (per AGENTS.md §5, run on a compute node, never the login node):

    sbatch --partition=aind_debug --cpus-per-task=1 --mem=8G --time=00:20:00 \
           --wrap "python extract_history_patterns.py --root /allen/aind/scratch/han.hou/tmp/rlgen --out <dir>"
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

# alias -> (wrapper alias == the analysis subdir, the r1/r8 baseline W&B run id)
BASELINES = {
    "ctt": ("ForagingCompareThreshold", "lmg1i9yd"),
    "bari": ("QLearning_L1F1_CK1_softmax", "bg3nzqz9"),
    "hattori": ("QLearning_L2F1_softmax", "unhmbrk4"),
}
WANTED_BLOCKS = ("config", "comparison", "subject_aggregate", "session_aggregate")
ENTITY_PROJECT = "AIND-disRNN/mice_data_scaling"
PRODUCED_BY = "variants/generative-rl-baseline/extract_history_patterns.py"
SOURCE_WRITTEN_BY = "variants/generative-rl-baseline/reanalyze_stats_only.py"


def lift_top_level_blocks(path: Path, wanted: tuple[str, ...]) -> dict:
    """Return {key: parsed value} for the named top-level keys, without loading the file.

    Relies on the source being `json.dumps(..., indent=2)`: every top-level key line starts
    with exactly two spaces and a quote, and a block ends where the next such line begins.
    """
    out: dict[str, str] = {}
    key, buf = None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith('  "') and '":' in line:
                if key is not None:
                    out[key] = "".join(buf)
                candidate = line.split('"')[1]
                if candidate in wanted:
                    key, buf = candidate, [line.split(":", 1)[1]]
                else:
                    key, buf = None, []
                continue
            if key is not None:
                buf.append(line)
    if key is not None:
        out[key] = "".join(buf)
    parsed = {}
    for k, raw in out.items():
        raw = raw.strip().rstrip(",")
        parsed[k] = json.loads(raw)
    missing = set(wanted) - set(parsed)
    if missing:
        raise ValueError(f"{path}: top-level blocks not found: {sorted(missing)}")
    return parsed


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the run_rl_rollout.py --out root holding <alias>/analysis/...")
    ap.add_argument("--out", required=True, help="output dir for the frozen JSONs")
    ap.add_argument("--summaries", default=None,
                    help="dir holding the committed <alias>_quantitative_summary.json, for "
                         "the cross-check; skipped when absent")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tz = ZoneInfo("America/Los_Angeles")

    for alias, (wrapper_alias, run_id) in BASELINES.items():
        src = root / alias / "analysis" / wrapper_alias / \
            "history_dependent_switch_stats_no_figures.json"
        print(f"=== {alias}: {src}", flush=True)
        stat = src.stat()
        blocks = lift_top_level_blocks(src, WANTED_BLOCKS)
        digest = sha256_of(src)

        panel = blocks["subject_aggregate"]["abstract"]["3"]
        if args.summaries:
            committed = json.load(
                open(Path(args.summaries) / f"{alias}_quantitative_summary.json")
            )["history_dependent"]["quantitative_summary"]["subject_mean"]["abstract"]["3"]
            for field in ("correlation", "rmse", "n_rows", "total_weight"):
                assert panel["summary"][field] == committed[field], (
                    f"{alias}: {field} disagrees with the committed summary "
                    f"({panel['summary'][field]} vs {committed[field]})"
                )
            print("  cross-check vs committed quantitative_summary: OK", flush=True)

        payload = {
            "_meta": {
                "produced_by": PRODUCED_BY,
                "produced_at_pt": datetime.datetime.now(tz).isoformat(timespec="seconds"),
                "source_file": str(src),
                "source_written_by": SOURCE_WRITTEN_BY,
                "source_bytes": stat.st_size,
                "source_mtime_pt": datetime.datetime.fromtimestamp(
                    stat.st_mtime, tz).isoformat(timespec="seconds"),
                "source_sha256": digest,
                "wandb_run_id": run_id,
                "wandb_entity_project": ENTITY_PROJECT,
                "wrapper_alias": wrapper_alias,
                "extracted_blocks": list(WANTED_BLOCKS),
                "note": (
                    "Frozen extract, not a re-computation: the rollout and the statistics "
                    "were produced once (run_rl_rollout.py 2026-07-14, "
                    "reanalyze_stats_only.py 2026-07-15) and are keyed here by "
                    "source_sha256. Subject-level per-mouse rows (subject_level, ~1.5 GB) "
                    "are deliberately NOT extracted."
                ),
            },
            **blocks,
        }
        dest = out / f"{alias}_history_patterns.json"
        dest.write_text(json.dumps(payload, indent=2))
        print(f"  -> {dest} ({dest.stat().st_size} bytes) | "
              f"abstract n=3: {len(panel['rows'])} rows, "
              f"r={panel['summary']['correlation']:.5f}", flush=True)

    os.system(f"chmod -R u+rwX {out}")


if __name__ == "__main__":
    main()
