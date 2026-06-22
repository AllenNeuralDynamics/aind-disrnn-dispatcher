"""Beaker launcher for RESUMABLE grid runs (one autoResume task per grid point).

This is the Option-1 alternative to ``launch_beaker.py``. Instead of creating a
W&B *sweep* drained by packed ``wandb agent`` replicas (where a preempted trial
is abandoned and re-run from scratch), it expands a grid sweep YAML into one
self-contained Beaker task per grid point, each running ``run_hpc`` with the
grid point's overrides baked in. Combined with:

  * Beaker ``preemptible: true`` (the server applies ``autoResume: true`` for
    preemptible/low-priority tasks — verified via ``beaker experiment spec``), so
    a preempted task is restarted as the *same* task with the *same* /results,
  * ``DISRNN_RESUMABLE_OUTPUT_DIR`` (run_hpc anchors outputs at a stable path
    instead of the per-run W&B dir), and
  * the trainer's full-state checkpoint/resume,

a preempted run continues from its last checkpoint instead of restarting. W&B
dashboard continuity across the restart is preserved by pinning a deterministic
``WANDB_RUN_ID`` per grid point with ``WANDB_RESUME=allow``.

Only ``method: grid`` sweeps are supported (the whole point of Option 1 — the
trial set must be enumerable up front; there is no sweep controller to ask).

Usage:
  python code/launch_beaker_resumable.py \
    --sweep code/beaker/sweep_gru_scaling.yaml \
    --experiment code/beaker/experiment_scaling.yaml \
    --workspace ai1/aind-dynamic-foraging-foundation-model
  python code/launch_beaker_resumable.py --sweep ... --experiment ... --no-submit
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path("/results")

BEAKER = shutil.which("beaker") or os.path.expanduser("~/.local/bin/beaker")

# Stable path (per task's own /results dataset) where run_hpc anchors outputs so
# autoResume restarts re-find their checkpoints. See run_hpc.py.
RESUMABLE_OUTPUT_DIR = "/results/run"
SWEEP_ARGS_PLACEHOLDER = "${args_no_hyphens}"


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


def _grid_points(parameters: dict) -> list[dict[str, object]]:
    """Cartesian product of a W&B-style grid ``parameters`` block.

    Each parameter is ``{values: [...]}`` (grid axis) or ``{value: x}`` (fixed).
    """
    axes: list[tuple[str, list]] = []
    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            sys.exit(f"[resumable] parameter {key!r} must be a mapping, got {spec!r}")
        if "values" in spec:
            axes.append((key, list(spec["values"])))
        elif "value" in spec:
            axes.append((key, [spec["value"]]))
        else:
            sys.exit(
                f"[resumable] parameter {key!r} needs 'values' or 'value' "
                f"(distribution-based search is not supported in grid mode)"
            )
    keys = [k for k, _ in axes]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(v for _, v in axes))]


def _override_str(key: str, value: object) -> str:
    """Render a Hydra override ``key=value``, matching W&B's grid formatting."""
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


def _run_id(group: str, overrides: list[str]) -> str:
    """Deterministic W&B run id for a grid point (stable across resume)."""
    digest = hashlib.sha1((group + "|" + "|".join(sorted(overrides))).encode()).hexdigest()
    return f"{group}-{digest[:8]}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _seattle_launch_id() -> str:
    """Readable, unique-per-launch id = Seattle local timestamp (see AGENTS §7/§8)."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")


def _study_variant(sweep_path: Path) -> tuple[str, str]:
    """Derive (study, variant) from a studies/<study>/variants/<variant>/sweep.yaml path.

    Falls back to the file stem when the path isn't under that layout, so the launcher
    still works for ad-hoc sweeps.
    """
    parts = sweep_path.resolve().parts
    study = variant = None
    if "studies" in parts and parts.index("studies") + 1 < len(parts):
        study = parts[parts.index("studies") + 1]
    if "variants" in parts and parts.index("variants") + 1 < len(parts):
        variant = parts[parts.index("variants") + 1]
    return study or "adhoc", variant or _slug(sweep_path.stem)


def build_spec(sweep_file: str, experiment_file: str, label: str | None = None) -> dict:
    """Expand a grid sweep + experiment template into a multi-task Beaker spec."""
    sweep = yaml.safe_load(Path(sweep_file).read_text())
    if sweep.get("method") != "grid":
        sys.exit(f"[resumable] only method: grid is supported, got {sweep.get('method')!r}")
    base_command = sweep.get("command")
    if not base_command or SWEEP_ARGS_PLACEHOLDER not in base_command:
        sys.exit(
            f"[resumable] sweep 'command' must include {SWEEP_ARGS_PLACEHOLDER} "
            f"(the swept overrides); got {base_command!r}"
        )
    grid = _grid_points(sweep.get("parameters", {}))
    if not grid:
        sys.exit("[resumable] sweep has no grid points")

    spec = yaml.safe_load(Path(experiment_file).read_text())
    template_task = spec["tasks"][0]
    # The bash + entrypoint.sh prefix from the template (drop its wandb-agent tail).
    entry_prefix = list(template_task["command"][:2])

    # Provenance / tracking (see AGENTS §8): one launch == one pseudo-sweep.
    #  - W&B group = "<variant>@<launch_id>" uniquely names THIS launch (distinguishes
    #    repeats; readable: variant -> study folder, launch_id -> Seattle time).
    #  - meta.* (study/variant/launch_id/label/config_hash) injected via DISRNN_META_*
    #    env; the wrapper stamps them into the run config alongside the platform-native
    #    BEAKER_EXPERIMENT_ID / BEAKER_JOB_ID / CO_COMPUTATION_ID.
    #  - launch_id is also folded into the run id, so every launch gets unique ids
    #    (distinguishes repeats AND avoids the deleted-id resync problem).
    sweep_path = Path(sweep_file)
    study, variant = _study_variant(sweep_path)
    launch_id = _seattle_launch_id()
    config_hash = hashlib.sha1(sweep_path.read_bytes()).hexdigest()[:8]
    group = f"{variant}@{launch_id}"            # W&B group = this launch (pseudo-sweep)
    id_base = _slug(f"{variant}-{launch_id}")    # slug-safe base for task names + run ids
    meta_env = [
        {"name": "DISRNN_META_STUDY", "value": study},
        {"name": "DISRNN_META_VARIANT", "value": variant},
        {"name": "DISRNN_META_LAUNCH_ID", "value": launch_id},
        {"name": "DISRNN_META_CONFIG_HASH", "value": config_hash},
    ]
    if label:
        meta_env.append({"name": "DISRNN_META_LABEL", "value": label})
    print(f"[resumable] study={study} variant={variant} launch_id={launch_id} "
          f"group={group} config_hash={config_hash}")

    tasks = []
    for index, point in enumerate(grid):
        overrides = [_override_str(k, v) for k, v in point.items()]
        # run_hpc command = sweep base command with the placeholder expanded.
        run_cmd: list = []
        for token in base_command:
            if token == SWEEP_ARGS_PLACEHOLDER:
                run_cmd.extend(overrides)
            else:
                run_cmd.append(token)

        task = copy.deepcopy(template_task)
        task.pop("replicas", None)
        task["name"] = f"{id_base}-{index:03d}"
        task["command"] = entry_prefix + run_cmd
        # Preemptible + low priority => Beaker applies autoResume (verified via
        # `beaker experiment spec`); the restart re-uses this task's /results.
        task["context"] = {"priority": "low", "preemptible": True}

        managed = {
            "DISRNN_RESUMABLE_OUTPUT_DIR", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP",
            "DISRNN_META_STUDY", "DISRNN_META_VARIANT", "DISRNN_META_LAUNCH_ID",
            "DISRNN_META_CONFIG_HASH", "DISRNN_META_LABEL",
        }
        env = [e for e in task.get("envVars", []) if e.get("name") not in managed]
        env.extend([
            {"name": "DISRNN_RESUMABLE_OUTPUT_DIR", "value": RESUMABLE_OUTPUT_DIR},
            {"name": "WANDB_RUN_ID", "value": _run_id(id_base, overrides)},
            {"name": "WANDB_RESUME", "value": "allow"},
            {"name": "WANDB_RUN_GROUP", "value": group},
            *meta_env,
        ])
        task["envVars"] = env
        tasks.append(task)

    spec["tasks"] = tasks
    spec["description"] = f"{group} — {len(tasks)} resumable grid tasks (autoResume)"
    return spec


def submit(spec: dict, workspace: str, output_dir: Path) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = output_dir / "experiment_resumable_submitted.yaml"
    rendered.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"[resumable] submitting {rendered.name} ({len(spec['tasks'])} tasks) to {workspace}")
    out = subprocess.run(
        [BEAKER, "experiment", "create", "-w", workspace, str(rendered)],
        capture_output=True, text=True,
    )
    print(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit(f"[resumable] `beaker experiment create` failed (exit {out.returncode})")
    m = re.search(r"Experiment\s+(\S+)\s+submitted", out.stdout + out.stderr)
    return m.group(1) if m else None


def save_record(sweep_file: str, experiment_id: str | None, n_tasks: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_file, output_dir / Path(sweep_file).name)
    record = {
        "mode": "resumable-grid",
        "n_tasks": n_tasks,
        "experiment_id": experiment_id,
        "experiment_url": f"https://beaker.org/ex/{experiment_id}" if experiment_id else None,
        "dispatcher_commit": _git_sha(REPO_ROOT),
        "sweep_file": str(sweep_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "beaker_resumable.json").write_text(json.dumps(record, indent=2))
    print(f"[resumable] saved record:\n{json.dumps(record, indent=2)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Launch a grid as one autoResume Beaker task per grid point."
    )
    p.add_argument("--sweep", default=str(REPO_ROOT / "code/beaker/sweep_gru_scaling.yaml"))
    p.add_argument("--experiment", default=str(REPO_ROOT / "code/beaker/experiment_scaling.yaml"),
                   help="experiment template providing image/cluster/resources/refs/secret")
    p.add_argument("--workspace", default="ai1/aind-dynamic-foraging-foundation-model")
    p.add_argument("--no-submit", action="store_true",
                   help="render the spec + record only; don't submit to Beaker")
    p.add_argument("--output-dir", default=str(RESULTS))
    p.add_argument("--label", default=None,
                   help="optional human label for this launch (stamped to W&B meta.label)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    spec = build_spec(args.sweep, args.experiment, label=args.label)
    n_tasks = len(spec["tasks"])
    print(f"[resumable] expanded grid into {n_tasks} tasks")

    experiment_id = None
    if args.no_submit:
        rendered = output_dir / "experiment_resumable_submitted.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(f"[resumable] --no-submit set; wrote {rendered}")
    else:
        experiment_id = submit(spec, args.workspace, output_dir)

    save_record(args.sweep, experiment_id, n_tasks, output_dir)
    print("[resumable] done")


if __name__ == "__main__":
    main()
