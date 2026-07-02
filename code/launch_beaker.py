"""Beaker control-plane launcher (dispatcher -> wrapper, AI Hub).

This is the dispatcher's reproducible-run entry for the AI Hub / Beaker path — the
analog of `generate_jobs.py` for the Code Ocean compute path. Running it (the CO
"Reproducible Run", via `code/run`) performs the dispatcher -> wrapper hand-off:

  1. create a W&B sweep from a sweep spec (default: code/beaker/sweep_mvp.yaml),
  2. save a reproducibility record to /results — the sweep YAML + SWEEP_ID +
     dispatcher commit (so the whole sweep is reproducible; CO persists /results
     under a CO_COMPUTATION_ID, which each Beaker run stamps into its W&B config),
  3. render the Beaker experiment spec with the SWEEP_ID, and
  4. submit it to Beaker.

Sweep creation and Beaker submission both go through code/beaker_client.py (GraphQL
+ beaker-py) rather than the `wandb`/`beaker` CLIs, so this SAME script runs
unmodified whether invoked on Allen HPC / Code Ocean or from the Claude Science
sandbox (Mac-based orchestration; see docs/claude-science-workflow.md) — see
beaker_client.py's docstring for why the CLI-based version couldn't do that.

The W&B key reaches the Beaker job via the Beaker secret referenced in the
experiment spec (see code/beaker/README.md) — it is not handled here.

Usage:
  python code/launch_beaker.py
  python code/launch_beaker.py --no-submit          # create the sweep + record only
  python code/launch_beaker.py --sweep <file> --experiment <file> --workspace <ws>
  python code/launch_beaker.py --output-dir ./out   # run outside Code Ocean (no /results)
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Reuse the SAME provenance helpers as the resumable launcher so the two routes can't
# drift (group naming, study/variant derivation, launch id). Safe to import: the sibling
# module guards execution behind `if __name__ == "__main__"`.
from launch_beaker_resumable import _seattle_launch_id, _study_variant

# Library-only clients (GraphQL for W&B, beaker-py for Beaker) -- no `wandb`/`beaker`
# CLI dependency; see code/beaker_client.py docstring for why.
from beaker_client import create_wandb_sweep, submit_beaker_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent  # the dispatcher capsule root (a git repo)
RESULTS = Path("/results")


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
    """Create a W&B sweep and return the full sweep path (entity/project/id).

    Uses the GraphQL API directly (see beaker_client.create_wandb_sweep) rather than
    shelling out to `wandb sweep`, which also works but spawns a local `wandb-core`
    helper service unnecessarily for a one-shot create call.
    """
    print(f"[launch_beaker] creating W&B sweep from {sweep_file}")
    try:
        sweep_id = create_wandb_sweep(sweep_file)
    except Exception as exc:
        sys.exit(f"[launch_beaker] W&B sweep creation failed: {exc}")
    print(f"[launch_beaker] SWEEP_ID = {sweep_id}")
    return sweep_id


def _render_experiment(experiment_file: str, sweep_id: str, group: str, meta_env: list) -> dict:
    """Render <SWEEP_ID> into the experiment spec and inject provenance env.

    Injects WANDB_RUN_GROUP (= <variant>@<launch_id>, consistent with the resumable
    route) + DISRNN_META_* into every task, so the wrapper's start_wandb_run stamps the
    portable meta.* alongside the native sweep + Beaker/CO ids. Pure (no I/O / network)
    so it's unit-testable.
    """
    spec = yaml.safe_load(Path(experiment_file).read_text().replace("<SWEEP_ID>", sweep_id))
    managed = {
        "WANDB_RUN_GROUP", "DISRNN_META_STUDY", "DISRNN_META_VARIANT",
        "DISRNN_META_LAUNCH_ID", "DISRNN_META_CONFIG_HASH", "DISRNN_META_LABEL",
        "DISRNN_META_NOTE",
    }
    for task in spec.get("tasks", []):
        env = [e for e in task.get("envVars", []) if e.get("name") not in managed]
        env.extend([{"name": "WANDB_RUN_GROUP", "value": group}, *meta_env])
        task["envVars"] = env
    return spec


def submit_experiment(
    experiment_file: str, sweep_id: str, workspace: str,
    group: str, meta_env: list, output_dir: Path = RESULTS,
) -> str | None:
    """Render <SWEEP_ID> + provenance into the experiment spec and `beaker experiment create`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = output_dir / "experiment_submitted.yaml"
    rendered.write_text(yaml.safe_dump(_render_experiment(experiment_file, sweep_id, group, meta_env), sort_keys=False))
    print(f"[launch_beaker] submitting {rendered.name} to {workspace}")
    try:
        experiment_id = submit_beaker_experiment(str(rendered), workspace)
    except Exception as exc:
        sys.exit(f"[launch_beaker] Beaker experiment submission failed: {exc}")
    print(f"[launch_beaker] Experiment {experiment_id} submitted")
    return experiment_id


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
    p.add_argument("--label", default=None,
                   help="optional human label for this launch (stamped to W&B meta.label)")
    p.add_argument("--note", default=None,
                   help="free-text background + what we want to learn (stamped to W&B meta.note)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    # Provenance (consistent with the resumable route — see AGENTS §8): one launch ==
    # one pseudo-sweep. Here the native W&B sweep is the platform-native launch id; we
    # still stamp the portable group + meta.* so both routes look identical in W&B.
    study, variant = _study_variant(Path(args.sweep))
    launch_id = _seattle_launch_id()
    config_hash = hashlib.sha1(Path(args.sweep).read_bytes()).hexdigest()[:8]
    group = f"{variant}@{launch_id}"
    meta_env = [
        {"name": "DISRNN_META_STUDY", "value": study},
        {"name": "DISRNN_META_VARIANT", "value": variant},
        {"name": "DISRNN_META_LAUNCH_ID", "value": launch_id},
        {"name": "DISRNN_META_CONFIG_HASH", "value": config_hash},
    ]
    if args.label:
        meta_env.append({"name": "DISRNN_META_LABEL", "value": args.label})
    if args.note:
        meta_env.append({"name": "DISRNN_META_NOTE", "value": args.note})
    print(f"[launch_beaker] study={study} variant={variant} launch_id={launch_id} "
          f"group={group} config_hash={config_hash}")

    sweep_id = create_sweep(args.sweep)
    print(f"[launch_beaker] SWEEP_ID = {sweep_id}")

    experiment_id = None
    if args.no_submit:
        rendered = output_dir / "experiment_submitted.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered.write_text(yaml.safe_dump(_render_experiment(args.experiment, sweep_id, group, meta_env), sort_keys=False))
        print(f"[launch_beaker] --no-submit set; wrote {rendered} (not submitted)")
    else:
        experiment_id = submit_experiment(args.experiment, sweep_id, args.workspace, group, meta_env, output_dir)

    save_record(args.sweep, sweep_id, experiment_id, output_dir)
    print("[launch_beaker] done")


if __name__ == "__main__":
    main()
