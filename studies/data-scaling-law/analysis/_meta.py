"""Provenance helper for analysis JSON outputs.

Per docs/posthoc-analysis.md "JSON output contract": every JSON written by an
analysis script must carry an opening `_meta` block. This helper centralises
the construction so producer scripts stay terse and the schema stays uniform.

Example:
    from _meta import build_meta
    out = {"_meta": build_meta("analysis/nxd_scaling.py", WANDB_GROUPS), ...}
    out_json.write_text(json.dumps(out, indent=2))

Times stamped in America/Los_Angeles per AGENTS.md section 7.
"""
from __future__ import annotations

import datetime
import re
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo


def _git_sha() -> str | None:
    """Best-effort `git rev-parse HEAD`; returns None outside a git checkout."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=Path(__file__).parent,
        ).strip()
        return sha or None
    except Exception:
        return None


def _wrapper_sha() -> str | None:
    """Wrapper commit pinned in the study's environment.lock, or None.

    This is the version of aind-disrnn-wrapper whose post_training_analysis
    produced the W&B keys these analyses read (the producer side of the
    contract in analysis/wandb_keys.py)."""
    lock = Path(__file__).resolve().parents[1] / "environment.lock"
    try:
        m = re.search(r"aind-disrnn-wrapper\.git@([0-9a-f]{7,40})", lock.read_text())
        return m.group(1) if m else None
    except Exception:
        return None


def build_meta(produced_by: str, wandb_groups: list[str]) -> dict:
    """Return a _meta dict per the JSON output contract.

    Args:
        produced_by: path to the producer script, relative to the study root
            (e.g. ``"analysis/nxd_scaling.py"``).
        wandb_groups: W&B group names this run pulled data from.
    """
    now_pt = datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    return {
        "produced_by": produced_by,
        "produced_at_pt": now_pt.isoformat(timespec="seconds"),
        "dispatcher_git_sha": _git_sha(),
        "wrapper_git_sha": _wrapper_sha(),
        "wandb_groups": list(wandb_groups),
    }
