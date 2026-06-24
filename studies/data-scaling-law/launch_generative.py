"""2nd-order (generative-behavioral) validation of data-scaling pretrain runs.

For each per-task training run in a SOURCE Beaker experiment (the data-scaling
pretrain phase), this launches one OFFLINE Beaker task that:

  * mounts that source task's result dataset at ``/prior`` (read-only); the
    trained run lives under ``/prior/run`` (``inputs.yaml`` +
    ``outputs/checkpoints/index.json`` + ``subject_index_map.json``),
  * rolls the trained model out as a generative agent and compares its behavior
    to the real animals via ``run_analysis generative`` — switch-triggered and
    history-dependent switch statistics, model-vs-animal deltas, and figures,
  * logs the key model-vs-animal scalar metrics + figures to a SEPARATE W&B
    group ``generative-<variant>@<launch_id>`` (project mice_data_scaling).

Split note: these are MULTISUBJECT (seen-subject) runs. ``resolve_model_run``
only supports ``split=train`` for multisubject runs, so the rollout uses
``--split train`` and breaks results down by session partition via
``--session-partitions train eval combined`` (the per-partition ``eval`` folder
is the held-out-session generative match; ``train``/``combined`` give context).

The ``generative`` analysis path has no built-in W&B logging, so each task runs
the analysis then a small in-container python step that reads the written
``model_vs_animal_quantitative_summary.json`` (per partition) and logs scalars
(mae / rmse / bias / correlation, plus weighted variants) + the figures to W&B.
WANDB_* group/project/entity ride as env; per-task source D/seed/meta are logged
to ``wandb.config``.

Usage:
  # validate a single task (subject_ratio=0.016 seed=0):
  .venv/bin/python studies/data-scaling-law/launch_generative.py \
    --source-exp 01KVQ7EJ3C5YJ8FJVNJB8C8N36 --variant v1 \
    --only-ratio 0.016 --only-seed 0
  # mass launch all tasks of a source exp:
  .venv/bin/python studies/data-scaling-law/launch_generative.py \
    --source-exp <EXP> --variant <name>
  # render only, don't submit:
  .venv/bin/python ... --no-submit
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pinned refs (see task context).
WRAPPER_REF = "916d3b497cfa866e14d1f041d1316f70c706f6e1"  # study: partition fix + parallel stats + smaller bootstrap
DISPATCHER_REF = "study/data-scaling-law"
IMAGE = "han-hou/disrnn-wrapper-pck-integration"
WANDB_PROJECT = "mice_data_scaling"
WANDB_ENTITY = "AIND-disRNN"

# Generative-rollout defaults (seen-subject multisubject runs).
MODEL_DIR = "/prior/run"
SPLIT = "train"
CHECKPOINT_POLICY = "best_eval"
ROLLOUT_MODE = "curriculum_matched"
N_ROLLOUTS_PER_SESSION = 1
WINDOW_SIZE = 10
SESSION_PARTITIONS = ["train", "eval", "combined"]
RESULTS_OUT = "/results/generative"

ENTRYPOINT = "/workspace/aind-disrnn-wrapper/beaker/entrypoint.sh"

BEAKER = shutil.which("beaker") or "/home/han/.local/bin/beaker"


def _seattle_launch_id() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")


def _git_sha(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def enumerate_source_tasks(source_exp: str) -> list[dict]:
    """Return [{result_dataset, subject_ratio, seed, job_name}] for each source task."""
    out = subprocess.run(
        [BEAKER, "experiment", "get", source_exp, "--format", "json"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        sys.exit(f"[generative] `beaker experiment get {source_exp}` failed:\n{out.stderr}")
    data = json.loads(out.stdout)
    exp = data[0] if isinstance(data, list) else data
    rows = []
    for job in exp.get("jobs", []):
        result = (job.get("result") or {}).get("beaker") or (
            (job.get("execution", {}).get("result") or {}).get("beaker")
        )
        cmd = job.get("execution", {}).get("spec", {}).get("command") or []
        cmds = " ".join(str(c) for c in cmd)
        m_ratio = re.search(r"subject_ratio=([0-9.eE+-]+)", cmds)
        m_seed = re.search(r"(?:^| )seed=([0-9]+)", cmds)
        if result is None or m_ratio is None or m_seed is None:
            continue
        rows.append({
            "result_dataset": result,
            "subject_ratio": m_ratio.group(1),
            "seed": m_seed.group(1),
            "job_name": job.get("name") or job.get("execution", {}).get("spec", {}).get("name"),
        })
    return rows


def _shq(s: str) -> str:
    """Single-quote a string for embedding in a bash -c command list element."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _wandb_log_py(group: str, study: str, variant: str, launch_id: str,
                  subject_ratio: str, seed: str, source_dataset: str,
                  source_exp: str, results_out: str) -> str:
    """In-container python that logs the generative model-vs-animal summary to W&B.

    Reads the per-partition ``model_vs_animal_quantitative_summary.json`` written
    under ``results_out/<partition>/`` and logs flattened scalar metrics + the
    figures, with provenance ``meta`` in wandb.config. No-op gracefully if a
    partition is missing (e.g. when a partition had no sessions).
    """
    meta = {
        "study": study, "variant": variant, "launch_id": launch_id,
        "source_subject_ratio": subject_ratio, "source_seed": seed,
        "source_dataset": source_dataset, "source_exp": source_exp,
        "kind": "generative",
    }
    return (
        "import json,os,glob,wandb\n"
        f"root={results_out!r}\n"
        f"meta={meta!r}\n"
        f"name='generative-{variant}-d{subject_ratio}-s{seed}'\n"
        "wandb.init(project=os.environ.get('WANDB_PROJECT'),"
        "entity=os.environ.get('WANDB_ENTITY'),group=os.environ.get('WANDB_RUN_GROUP'),"
        "name=name,config={'meta':meta})\n"
        "parts=['train','eval','combined']\n"
        "_log={}\n"
        "for part in parts:\n"
        " f=os.path.join(root,part,'model_vs_animal_quantitative_summary.json')\n"
        " if not os.path.exists(f): continue\n"
        " stack=[([part],json.load(open(f)))]\n"
        " while stack:\n"
        "  pre,obj=stack.pop()\n"
        "  for k,v in obj.items():\n"
        "   if isinstance(v,dict): stack.append((pre+[str(k)],v))\n"
        "   elif isinstance(v,(int,float)) and not isinstance(v,bool): _log['/'.join(pre+[str(k)])]=v\n"
        "if _log: wandb.log(_log)\n"
        "for part in parts:\n"
        " fig=os.path.join(root,part,'figures')\n"
        " if not os.path.isdir(fig): continue\n"
        " imgs={part+'/'+os.path.splitext(os.path.basename(p))[0]:wandb.Image(p) for p in sorted(glob.glob(os.path.join(fig,'*.png')))}\n"
        " if imgs: wandb.log(imgs)\n"
        "wandb.finish()\n"
    )


def _inner_cmd(group: str, study: str, variant: str, launch_id: str,
               subject_ratio: str, seed: str, source_dataset: str,
               source_exp: str) -> str:
    """Run the generative analysis, then log the summary + figures to W&B."""
    gen = (
        "python -m run_analysis generative"
        f" --model-dir {MODEL_DIR}"
        f" --split {SPLIT}"
        f" --checkpoint-policy {CHECKPOINT_POLICY}"
        f" --rollout-mode {ROLLOUT_MODE}"
        f" --n-rollouts-per-session {N_ROLLOUTS_PER_SESSION}"
        f" --window-size {WINDOW_SIZE}"
        f" --session-partitions {' '.join(SESSION_PARTITIONS)}"
        f" --output-dir {RESULTS_OUT}"
    )
    log_py = _wandb_log_py(
        group=group, study=study, variant=variant, launch_id=launch_id,
        subject_ratio=subject_ratio, seed=seed, source_dataset=source_dataset,
        source_exp=source_exp, results_out=RESULTS_OUT,
    )
    return f"{gen} && python -c {_shq(log_py)}"


def build_spec(source_exp: str, variant: str, tasks: list[dict], cluster: str,
               launch_id: str, wrapper_ref: str) -> tuple[dict, str]:
    study = "data-scaling-law"
    group = f"generative-{variant}@{launch_id}"
    spec_tasks = []
    for row in tasks:
        inner = _inner_cmd(
            group=group, study=study, variant=variant, launch_id=launch_id,
            subject_ratio=row["subject_ratio"], seed=row["seed"],
            source_dataset=row["result_dataset"], source_exp=source_exp,
        )
        name = f"generative-{variant}-d{row['subject_ratio']}-s{row['seed']}"
        name = re.sub(r"[^a-zA-Z0-9-]+", "-", name).strip("-")[:120]
        spec_tasks.append({
            "name": name,
            "image": {"beaker": IMAGE},
            "command": ["bash", ENTRYPOINT, "bash", "-lc", inner],
            "context": {"priority": "low", "preemptible": True},
            "constraints": {"cluster": [cluster]},
            # 90GiB / 12 CPU == ONE L40s GPU bundle -> exactly 1 GPU (AGENTS.md §10).
            "resources": {"gpuCount": 1, "cpuCount": 12, "memory": "90GiB"},
            "datasets": [
                {"mountPath": "/prior", "source": {"beaker": row["result_dataset"]}},
            ],
            "envVars": [
                {"name": "WANDB_API_KEY", "secret": "han-wandb-api-key"},
                {"name": "WANDB_PROJECT", "value": WANDB_PROJECT},
                {"name": "WANDB_ENTITY", "value": WANDB_ENTITY},
                {"name": "WANDB_RUN_GROUP", "value": group},
                {"name": "WRAPPER_REF", "value": wrapper_ref},
                {"name": "DISPATCHER_REF", "value": DISPATCHER_REF},
                # Fan the 3 independent session partitions (train/eval/combined)
                # across a spawn pool — the stats phase dominates high-D
                # wall-clock. 3 = one worker per partition (combined is the pole).
                {"name": "DISRNN_GENERATIVE_STATS_WORKERS", "value": "3"},
            ],
            "result": {"path": "/results"},
        })
    spec = {
        "version": "v2",
        "description": f"{group} — {len(spec_tasks)} generative-validation tasks from {source_exp}",
        "tasks": spec_tasks,
    }
    return spec, group


def submit(spec: dict, workspace: str, rendered_path: Path) -> str | None:
    rendered_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"[generative] submitting {rendered_path} ({len(spec['tasks'])} tasks) to {workspace}")
    out = subprocess.run(
        [BEAKER, "experiment", "create", "-w", workspace, str(rendered_path)],
        capture_output=True, text=True,
    )
    print(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit(f"[generative] `beaker experiment create` failed (exit {out.returncode})")
    m = re.search(r"(?:Experiment\s+|/ex/)(\S+?)(?:\s+submitted|\s|$)", out.stdout + out.stderr)
    return m.group(1) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-exp", required=True, help="Source training Beaker experiment id.")
    p.add_argument("--variant", required=True, help="e.g. v1 / v2 (group = generative-<variant>@<launch_id>).")
    p.add_argument("--cluster", default="ai1/octo-hub-aws-l40s")
    p.add_argument("--workspace", default="ai1/aind-dynamic-foraging-foundation-model")
    p.add_argument("--limit", type=int, default=None, help="Only the first N source tasks.")
    p.add_argument("--only-ratio", default=None, help="Filter to this subject_ratio.")
    p.add_argument("--only-seed", default=None, help="Filter to this seed.")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument("--wrapper-ref", default=WRAPPER_REF, help="Wrapper git ref the entrypoint checks out.")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = p.parse_args()

    all_tasks = enumerate_source_tasks(args.source_exp)
    print(f"[generative] source exp {args.source_exp}: {len(all_tasks)} tasks")
    tasks = all_tasks
    if args.only_ratio is not None:
        tasks = [t for t in tasks if abs(float(t["subject_ratio"]) - float(args.only_ratio)) < 1e-9]
    if args.only_seed is not None:
        tasks = [t for t in tasks if t["seed"] == str(args.only_seed)]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit("[generative] no source tasks matched the filters")
    print(f"[generative] selected {len(tasks)} task(s):")
    for t in tasks:
        print(f"    D={t['subject_ratio']} seed={t['seed']} dataset={t['result_dataset']}")

    launch_id = _seattle_launch_id()
    spec, group = build_spec(args.source_exp, args.variant, tasks, args.cluster,
                             launch_id, args.wrapper_ref)
    record_stem = group.split("@")[0]  # generative-<variant>

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = output_dir / f"{record_stem}_{launch_id}.yaml"

    experiment_id = None
    if args.no_submit:
        rendered.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(f"[generative] --no-submit; wrote {rendered}")
    else:
        experiment_id = submit(spec, args.workspace, rendered)

    record = {
        "kind": "generative",
        "source_exp": args.source_exp,
        "variant": args.variant,
        "launch_id": launch_id,
        "wandb_group": group,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "wrapper_ref": args.wrapper_ref,
        "dispatcher_ref": DISPATCHER_REF,
        "cluster": args.cluster,
        "experiment_id": experiment_id,
        "experiment_url": f"https://beaker.org/ex/{experiment_id}" if experiment_id else None,
        "n_tasks": len(tasks),
        "tasks": tasks,
        "dispatcher_commit": _git_sha(REPO_ROOT),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rendered_spec": str(rendered),
    }
    rec_path = output_dir / f"{record_stem}_{launch_id}.json"
    rec_path.write_text(json.dumps(record, indent=2))
    print(f"[generative] saved record: {rec_path}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
