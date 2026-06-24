#!/usr/bin/env python
"""Watch the D=30 N x D gap-fill and regenerate report artifacts when done."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import wandb

ROOT = Path(__file__).resolve().parents[3]
PROJECT = "AIND-disRNN/mice_data_scaling"
GROUP = "nxd-grid@20260624-141106"
EXPECTED_HIDDEN = {16, 64, 256}
EXPECTED_RATIO = 0.049
EXPECTED_SEEDS = {0, 1, 2}
METRICS = ("heldout/final/eval_likelihood", "heldout/eval_likelihood")
PT = ZoneInfo("America/Los_Angeles")
TERMINAL_BAD_STATES = {"failed", "crashed", "killed"}


def stamp() -> str:
    return datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")


def nested(config: dict, path: tuple[str, ...], default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def run_key(run) -> tuple[int | None, float | None, int | None]:
    config = run.config or {}
    h = nested(config, ("model", "architecture", "hidden_size"), config.get("model.architecture.hidden_size"))
    ratio = nested(config, ("data", "subject_ratio"), config.get("data.subject_ratio"))
    seed = nested(config, ("data", "subject_sample_seed"), config.get("seed"))
    return (
        None if h is None else int(h),
        None if ratio is None else round(float(ratio), 3),
        None if seed is None else int(seed),
    )


def metric_ready(run) -> bool:
    summary = run.summary or {}
    return any(summary.get(name) is not None for name in METRICS)


def poll(api: wandb.Api) -> tuple[bool, str]:
    runs = list(api.runs(PROJECT, filters={"group": GROUP}))
    states = Counter(r.state for r in runs)
    expected = {(h, round(EXPECTED_RATIO, 3), seed) for h in EXPECTED_HIDDEN for seed in EXPECTED_SEEDS}
    seen = {run_key(r): r for r in runs if run_key(r) in expected}
    missing = sorted(expected - set(seen))
    bad = [r for r in runs if r.state in TERMINAL_BAD_STATES]

    if bad:
        detail = ", ".join(f"{r.id}:{r.state}" for r in bad)
        raise RuntimeError(f"terminal failed D30 runs: {detail}")

    finished = [r for r in seen.values() if r.state == "finished"]
    finished_with_metric = [r for r in finished if metric_ready(r)]
    msg = (
        f"runs={len(runs)} expected_seen={len(seen)}/9 states={dict(states)} "
        f"finished_with_metric={len(finished_with_metric)}/9"
    )
    if missing:
        msg += f" missing={missing}"
    return len(seen) == 9 and len(finished_with_metric) == 9, msg


def regenerate() -> None:
    commands = [
        [sys.executable, "studies/data-scaling-law/analysis/nxd_scaling.py"],
        [sys.executable, "studies/data-scaling-law/analysis/update_final_report_nxd.py"],
    ]
    for command in commands:
        print(f"{stamp()} running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--once", action="store_true", help="Poll once, then exit without regenerating.")
    args = parser.parse_args()

    deadline = datetime.now(PT) + timedelta(hours=args.timeout_hours)
    api = wandb.Api(timeout=60)
    print(f"{stamp()} watching {PROJECT} group {GROUP}", flush=True)
    while True:
        ready, msg = poll(api)
        print(f"{stamp()} {msg}", flush=True)
        if args.once:
            return
        if ready:
            regenerate()
            print(f"{stamp()} complete", flush=True)
            return
        if datetime.now(PT) >= deadline:
            raise TimeoutError(f"timed out before all D30 runs finished: {msg}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
