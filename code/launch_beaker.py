"""Beaker control-plane launcher (dispatcher -> wrapper, AI Hub).

This is the dispatcher's reproducible-run entry for the AI Hub / Beaker path — the
analog of `generate_jobs.py` for the Code Ocean compute path. Running it (the CO
"Reproducible Run", via `code/run`) performs the dispatcher -> wrapper hand-off:

  1. create a W&B sweep from a sweep spec (default: code/beaker/sweep_mvp.yaml),
  2. save a reproducibility record to /results — the sweep YAML + SWEEP_ID +
     dispatcher commit (so the whole sweep is reproducible; CO persists /results
     under a CO_COMPUTATION_ID, which each Beaker run stamps into its W&B config),
  3. render the Beaker experiment spec with the SWEEP_ID, and
  4. submit it with `beaker experiment create`.

The W&B key reaches the Beaker job via the Beaker secret referenced in the
experiment spec (see code/beaker/README.md) — it is not handled here.

Usage:
  python code/launch_beaker.py
  python code/launch_beaker.py --no-submit          # create the sweep + record only
  python code/launch_beaker.py --sweep <file> --experiment <file> --workspace <ws>
  python code/launch_beaker.py --output-dir ./out   # run outside Code Ocean (no /results)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # the dispatcher capsule root (a git repo)
RESULTS = Path("/results")

# wandb is on PATH; beaker is installed to ~/.local/bin by environment/postInstall
# and may not be on PATH in a non-login shell, so fall back to the known location.
WANDB = shutil.which("wandb") or "wandb"
BEAKER = shutil.which("beaker") or os.path.expanduser("~/.local/bin/beaker")


def _git_sha(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def create_sweep(sweep_file: str) -> str:
    """Run `wandb sweep` and return the full sweep path (entity/project/id)."""
    print(f"[launch_beaker] creating W&B sweep from {sweep_file}")
    out = subprocess.run([WANDB, "sweep", sweep_file], capture_output=True, text=True)
    combined = out.stdout + out.stderr
    print(combined)
    if out.returncode != 0:
        sys.exit(f"[launch_beaker] `wandb sweep` failed (exit {out.returncode})")
    m = re.search(r"wandb agent\s+(\S+)", combined)
    if not m:
        sys.exit("[launch_beaker] could not parse SWEEP_ID from wandb output")
    return m.group(1)


def submit_experiment(
    experiment_file: str, sweep_id: str, workspace: str, output_dir: Path = RESULTS
) -> str | None:
    """Render <SWEEP_ID> into the experiment spec and `beaker experiment create`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = output_dir / "experiment_submitted.yaml"
    rendered.write_text(Path(experiment_file).read_text().replace("<SWEEP_ID>", sweep_id))
    print(f"[launch_beaker] submitting {rendered.name} to {workspace}")
    out = subprocess.run(
        [BEAKER, "experiment", "create", "-w", workspace, str(rendered)],
        capture_output=True, text=True,
    )
    print(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit(f"[launch_beaker] `beaker experiment create` failed (exit {out.returncode})")
    m = re.search(r"Experiment\s+(\S+)\s+submitted", out.stdout + out.stderr)
    return m.group(1) if m else None


def save_record(
    sweep_file: str, sweep_id: str, experiment_id: str | None, output_dir: Path = RESULTS
) -> None:
    """Persist the sweep YAML + IDs + commit to output_dir for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_file, output_dir / Path(sweep_file).name)
    entity_project, _, sweep_short = sweep_id.rpartition("/")
    record = {
        "sweep_id": sweep_id,
        "sweep_url": f"https://wandb.ai/{entity_project}/sweeps/{sweep_short}",
        "experiment_id": experiment_id,
        "experiment_url": (
            f"https://beaker.org/ex/{experiment_id}" if experiment_id else None
        ),
        "dispatcher_commit": _git_sha(REPO_ROOT),
        "sweep_file": str(sweep_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "beaker_sweep.json").write_text(json.dumps(record, indent=2))
    print(f"[launch_beaker] saved reproducibility record:\n{json.dumps(record, indent=2)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Create a W&B sweep and launch it on Beaker.")
    p.add_argument("--sweep", default=str(REPO_ROOT / "code/beaker/sweep_mvp.yaml"))
    p.add_argument("--experiment", default=str(REPO_ROOT / "code/beaker/experiment_mvp.yaml"))
    p.add_argument("--workspace", default="ai1/aind-dynamic-foraging-foundation-model")
    p.add_argument("--no-submit", action="store_true",
                   help="create the sweep + reproducibility record only; don't submit to Beaker")
    p.add_argument("--output-dir", default=str(RESULTS),
                   help="directory for the rendered spec + reproducibility record "
                        "(default: /results, the Code Ocean mount). Set this to run outside CO.")
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    sweep_id = create_sweep(args.sweep)
    print(f"[launch_beaker] SWEEP_ID = {sweep_id}")

    experiment_id = None
    if args.no_submit:
        print("[launch_beaker] --no-submit set; skipping `beaker experiment create`")
    else:
        experiment_id = submit_experiment(args.experiment, sweep_id, args.workspace, output_dir)

    save_record(args.sweep, sweep_id, experiment_id, output_dir)
    print("[launch_beaker] done")


if __name__ == "__main__":
    main()
