#!/usr/bin/env python
"""Freeze the GRU side of r9's history-pattern scatter, straight out of Beaker.

WHY THIS EXISTS. r9's generative tasks write `history_dependent_switch_stats.json` per
session partition, but `launch_generative.py`'s in-container W&B step uploads only the
flattened numeric leaves of `model_vs_animal_quantitative_summary.json` plus `figures/*.png`.
The per-pattern rows (`animal_mean` / `simulated_mean` ± SEM per history pattern) are
therefore NOT in W&B. They are in each task's Beaker result dataset, because the task's
`--output-dir` is `/results`.

The `combined` partition's file is ~1.55 GB, so this streams it over HTTPS and lifts out the
four top-level blocks the scatter needs (top-level keys sit at two-space indent under
`json.dumps(indent=2)`, so a block runs from its key line to the next such line). 1.55 GB
never lands on disk and never enters memory; the output is ~200 KB per task.

⚠️  PROVENANCE CAVEAT, carried into every output's `_meta`. These rollouts predate wrapper
PR #60, so off-curriculum sessions were simulated as a default uncoupled-baiting task rather
than the family the mouse actually ran. The RL rollouts these panels get compared against are
post-#60. A figure built from both sides is a PROVISIONAL look, not a result; the fix is
re-running r9's generative rollout on current wrapper main (tracked in study 05's r4).

Requires BEAKER_TOKEN in the environment. Network: beaker.org (API) + data.beaker.org
(dataset storage).

Usage:
  python analysis/fetch_gru_history_patterns.py --out analysis/gru_history_patterns
  python analysis/fetch_gru_history_patterns.py --tasks v2_d614_s0 --partitions combined eval
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

WANTED_BLOCKS = ("config", "comparison", "subject_aggregate", "session_aggregate")
BEAKER_API = "https://beaker.org/api/v3"
ENTITY_PROJECT = "AIND-disRNN/mice_data_scaling"
PRODUCED_BY = "analysis/fetch_gru_history_patterns.py"

# Resolved 2026-08-31 by walking the Beaker workspace
# ai1/aind-dynamic-foraging-foundation-model: the r9 launch records for these two groups were
# never committed, so the experiment ids are recorded here instead. D = subject_ratio x 614;
# every task in both groups is hidden_size=128 (the N x D grid is a separate experiment).
TASKS = {
    "v1_d614_s0": {
        "dataset": "01KVVJPJMHZH5D8JREBF4X90C7",
        "experiment": "01KVVJPGA55QSMHJMSRFA5WMHK",
        "job": "generative-v1-d1-0-s0",
        "wandb_group": "generative-v1@20260623-180747",
        "wandb_run_id": "yqjbjiq5",
        "source_exp": "01KVQ7EJ3C5YJ8FJVNJB8C8N36",
        "source_dataset": "01KVQ7EKECRXB4YVWQKK0N73KM",
        "variant": "v1",
        "session_conditioning": "declared (scalar) but never activated — no session pretrain/warmup schedule",
        "hidden_size": 128,
        "n_train_mice": 614,
        "seed": 0,
    },
    "v2_d614_s0": {
        "dataset": "01KVVJPN3M6S3QA8KWF8SDMC1G",
        "experiment": "01KVVJPKT31HTFWA4Y9SRY141M",
        "job": "generative-v2-d1-0-s0",
        "wandb_group": "generative-v2@20260623-180750",
        "wandb_run_id": "bfdmcyfd",
        "source_exp": "01KVRMSAAJTRSJMFV5JT7JAP6X",
        "source_dataset": "01KVRMSBRV8QFHGS3XYRAS9P9A",
        "variant": "v2",
        "session_conditioning": "active — session_n_pretrain_steps=30000, session_n_warmup_steps=20000",
        "hidden_size": 128,
        "n_train_mice": 614,
        "seed": 0,
    },
}
WRAPPER_CAVEAT = (
    "Rollout predates wrapper PR #60: off-curriculum sessions were simulated as a default "
    "uncoupled-baiting task instead of the family the mouse ran. The RL baseline rollouts in "
    "05-disrnn-scaling-law/variants/generative-rl-baseline are post-#60. Treat any GRU-vs-RL "
    "figure built from this as provisional."
)


def beaker_token() -> str:
    """Return BEAKER_TOKEN, or exit with an actionable message rather than a KeyError."""
    token = os.environ.get("BEAKER_TOKEN")
    if not token:
        raise SystemExit(
            "BEAKER_TOKEN is not set. This script reads the generative tasks' Beaker result\n"
            "datasets directly. Get a token from https://beaker.org/user (or `beaker account "
            "token`)\nand export it:\n\n    export BEAKER_TOKEN=...\n"
        )
    return token


def storage_for(dataset_id: str) -> tuple[str, dict]:
    """Return (files_root_url, auth_headers) for a Beaker dataset's storage backend."""
    req = urllib.request.Request(
        f"{BEAKER_API}/datasets/{dataset_id}",
        headers={"Authorization": f"Bearer {beaker_token()}"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        meta = json.load(response)
    storage = meta["storage"]
    base = storage["address"].rstrip("/")
    # Paths must keep their literal slashes; %2F-encoding them 404s.
    return (f"{base}/datasets/{storage['id']}/files",
            {"Authorization": f"Bearer {storage['token']}"})


def stream_top_level_blocks(url: str, headers: dict, wanted: tuple[str, ...]) -> tuple[dict, str, int]:
    """Stream a giant indent=2 JSON file, returning ({block: value}, sha256, n_bytes).

    The file is never buffered whole: bytes are hashed as they arrive and only the wanted
    blocks' text is retained.
    """
    digest = hashlib.sha256()
    n_bytes = 0

    class _Counting(io.RawIOBase):
        def __init__(self, raw):
            self.raw = raw

        def readable(self):
            return True

        def readinto(self, buf):
            nonlocal n_bytes
            chunk = self.raw.read(len(buf))
            if not chunk:
                return 0
            digest.update(chunk)
            n_bytes += len(chunk)
            buf[: len(chunk)] = chunk
            return len(chunk)

    collected: dict[str, str] = {}
    key, parts = None, []
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=1800) as response:
        text = io.TextIOWrapper(io.BufferedReader(_Counting(response), buffer_size=1 << 22),
                               encoding="utf-8")
        for line in text:
            if line.startswith('  "') and '":' in line:
                if key is not None:
                    collected[key] = "".join(parts)
                candidate = line.split('"')[1]
                if candidate in wanted:
                    key, parts = candidate, [line.split(":", 1)[1]]
                else:
                    key, parts = None, []
                continue
            if key is not None:
                parts.append(line)
    if key is not None:
        collected[key] = "".join(parts)

    parsed = {}
    for name, raw in collected.items():
        raw = raw.strip().rstrip(",")
        parsed[name] = json.loads(raw)
    missing = set(wanted) - set(parsed)
    if missing:
        raise ValueError(f"{url}: blocks not found: {sorted(missing)}")
    return parsed, digest.hexdigest(), n_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "gru_history_patterns"))
    ap.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    ap.add_argument("--partitions", nargs="*", default=["combined"],
                    help="train | eval | combined (per-mouse SESSION splits, all 614 mice)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tz = ZoneInfo("America/Los_Angeles")

    for label in args.tasks:
        spec = TASKS[label]
        root, headers = storage_for(spec["dataset"])
        for partition in args.partitions:
            url = f"{root}/generative/{partition}/history_dependent_switch_stats.json"
            print(f"=== {label} / {partition}: streaming {url.rsplit('/files/', 1)[-1]}", flush=True)
            blocks, digest, n_bytes = stream_top_level_blocks(url, headers, WANTED_BLOCKS)
            panel = blocks["subject_aggregate"]["abstract"]["3"]
            payload = {
                "_meta": {
                    "produced_by": PRODUCED_BY,
                    "produced_at_pt": datetime.datetime.now(tz).isoformat(timespec="seconds"),
                    "model": "GRU",
                    "session_partition": partition,
                    "source_file": f"beaker://{spec['dataset']}/generative/{partition}/"
                                   "history_dependent_switch_stats.json",
                    "source_bytes": n_bytes,
                    "source_sha256": digest,
                    "beaker_dataset": spec["dataset"],
                    "beaker_experiment": spec["experiment"],
                    "beaker_job": spec["job"],
                    "wandb_entity_project": ENTITY_PROJECT,
                    "wandb_group": spec["wandb_group"],
                    "wandb_run_id": spec["wandb_run_id"],
                    "source_exp": spec["source_exp"],
                    "source_dataset": spec["source_dataset"],
                    "variant": spec["variant"],
                    "session_conditioning": spec["session_conditioning"],
                    "hidden_size": spec["hidden_size"],
                    "n_train_mice": spec["n_train_mice"],
                    "seed": spec["seed"],
                    "extracted_blocks": list(WANTED_BLOCKS),
                    "wrapper_caveat": WRAPPER_CAVEAT,
                },
                **blocks,
            }
            dest = out / f"{label}_{partition}_history_patterns.json"
            dest.write_text(json.dumps(payload, indent=2))
            print(f"  streamed {n_bytes/1e9:.2f} GB -> {dest.name} "
                  f"({dest.stat().st_size} B) | abstract n=3: {len(panel['rows'])} rows, "
                  f"r={panel['summary']['correlation']:.5f} rmse={panel['summary']['rmse']:.5f}",
                  flush=True)


if __name__ == "__main__":
    main()
