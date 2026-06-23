"""Re-run held-out-subject fine-tune OFFLINE from saved checkpoints (no retraining).

For each per-task training run in a SOURCE Beaker experiment (the data-scaling
pretrain phase), this launches one offline Beaker task that:

  * mounts that source task's result dataset at ``/prior`` (read-only),
  * runs the wrapper's ``run_analysis finetune`` against
    ``studies/data-scaling-law/heldout_offline.yaml`` (``source_run.model_dir:
    /prior/run`` — the training run lives under ``run/`` in the result dataset:
    ``run/inputs.yaml`` + ``run/outputs/checkpoints/index.json``),
  * fine-tunes 500 steps from the source ``best_eval`` checkpoint and writes
    ``per_subject_likelihood.json`` (added by wrapper commit 4f296807),
  * pins ``WRAPPER_REF`` to the per-subject-likelihood commit and ``DISPATCHER_REF``
    to the study branch so the entrypoint pulls fresh code/config before running.

W&B: each offline job logs to a SEPARATE group ``heldout-rerun-<variant>@<launch_id>``
(project mice_data_scaling) — it does NOT touch the original runs. The wrapper's
finetune path only starts a W&B run when the config's ``wandb`` block is non-empty,
and ``heldout_offline.yaml`` ships ``wandb: {}``. Rather than committing a config
edit, each task injects a populated ``wandb`` block (+ provenance ``meta``) into a
temp copy of the config at container start, then points ``finetune --config`` at it.
``WANDB_RUN_GROUP``/``WANDB_PROJECT``/``WANDB_ENTITY`` are passed as env (wandb reads
them natively); per-task source D/seed/run-id ride along in the injected ``meta``.

Usage:
  # validate a single task (subject_ratio=0.016 seed=0):
  .venv/bin/python studies/data-scaling-law/launch_heldout_rerun.py \
    --source-exp 01KVQ7EJ3C5YJ8FJVNJB8C8N36 --variant v1 \
    --only-ratio 0.016 --only-seed 0
  # mass launch all tasks of a source exp:
  .venv/bin/python studies/data-scaling-law/launch_heldout_rerun.py \
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
STUDY_DIR_REL = "studies/data-scaling-law"
CONFIG_REL = f"{STUDY_DIR_REL}/heldout_offline.yaml"

# Pinned refs (see task context).
WRAPPER_REF = "4f296807f4ea06f9a58afa4eeb0553c220db4726"  # adds per_subject_likelihood.json
DISPATCHER_REF = "study/data-scaling-law"
IMAGE = "han-hou/disrnn-wrapper-pck-integration"
WANDB_PROJECT = "mice_data_scaling"
WANDB_ENTITY = "AIND-disRNN"

# Container paths (entrypoint cd's into the wrapper's code/ dir).
DISPATCHER_DIR = "/workspace/aind-disrnn-dispatcher"
CONFIG_IN_CONTAINER = f"{DISPATCHER_DIR}/{CONFIG_REL}"
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
        sys.exit(f"[heldout-rerun] `beaker experiment get {source_exp}` failed:\n{out.stderr}")
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


def _inject_config_cmd(group: str, study: str, variant: str, launch_id: str,
                       subject_ratio: str, seed: str, source_dataset: str,
                       source_exp: str, config_in_container: str, kind: str) -> str:
    """A python one-liner (run in-container) that copies the held-out config,
    fills its empty ``wandb`` block + adds a ``meta`` block, and writes /tmp config.

    ``config_in_container`` is the in-container path to the source config
    (heldout_offline.yaml for adapted reruns, heldout_zeroshot.yaml for the
    no-adaptation zero-shot variant). ``kind`` tags ``meta.kind`` for provenance.

    The wrapper's finetune path only starts a W&B run when ``config['wandb']`` is
    truthy; populating it (and relying on native WANDB_* env for group/project) is
    the no-commit way to enable per-job W&B logging. The whole config (incl. meta)
    lands in wandb.config for provenance.
    """
    tmp_config = f"/tmp/{Path(config_in_container).name}"
    py = (
        "import yaml,os;"
        f"c=yaml.safe_load(open({config_in_container!r}));"
        "c['wandb']={'project':os.environ.get('WANDB_PROJECT'),"
        "'entity':os.environ.get('WANDB_ENTITY'),"
        "'group':os.environ.get('WANDB_RUN_GROUP')};"
        "c['meta']={"
        f"'study':{study!r},'variant':{variant!r},'launch_id':{launch_id!r},"
        f"'source_subject_ratio':{subject_ratio!r},'source_seed':{seed!r},"
        f"'source_dataset':{source_dataset!r},'source_exp':{source_exp!r},"
        f"'kind':{kind!r}}};"
        f"open({tmp_config!r},'w').write(yaml.safe_dump(c))"
    )
    return (
        f"python -c {_shq(py)} && "
        f"python -m run_analysis finetune --config {tmp_config}"
    )


def _shq(s: str) -> str:
    """Single-quote a string for embedding in a bash -c command list element."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def build_spec(source_exp: str, variant: str, tasks: list[dict], cluster: str,
               launch_id: str, config_rel: str, wrapper_ref: str) -> tuple[dict, str]:
    study = "data-scaling-law"
    # Zero-shot (no-adaptation) configs get a distinct group/kind so they sit beside
    # the adapted reruns in the same W&B project without colliding.
    is_zeroshot = "zeroshot" in Path(config_rel).stem.lower()
    group_kind = "heldout-zeroshot" if is_zeroshot else "heldout-rerun"
    meta_kind = "heldout-zeroshot" if is_zeroshot else "heldout-rerun-offline"
    config_in_container = f"{DISPATCHER_DIR}/{config_rel}"
    group = f"{group_kind}-{variant}@{launch_id}"
    spec_tasks = []
    for row in tasks:
        inner = _inject_config_cmd(
            group=group, study=study, variant=variant, launch_id=launch_id,
            subject_ratio=row["subject_ratio"], seed=row["seed"],
            source_dataset=row["result_dataset"], source_exp=source_exp,
            config_in_container=config_in_container, kind=meta_kind,
        )
        name = f"{group_kind}-{variant}-d{row['subject_ratio']}-s{row['seed']}"
        name = re.sub(r"[^a-zA-Z0-9-]+", "-", name).strip("-")[:120]
        spec_tasks.append({
            "name": name,
            "image": {"beaker": IMAGE},
            "command": ["bash", ENTRYPOINT, "bash", "-lc", inner],
            "context": {"priority": "low", "preemptible": True},
            "constraints": {"cluster": [cluster]},
            "resources": {"gpuCount": 1, "cpuCount": 16, "memory": "256GiB"},
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
            ],
            "result": {"path": "/results"},
        })
    spec = {
        "version": "v2",
        "description": f"{group} — {len(spec_tasks)} offline held-out tasks from {source_exp}",
        "tasks": spec_tasks,
    }
    return spec, group


def submit(spec: dict, workspace: str, rendered_path: Path) -> str | None:
    rendered_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"[heldout-rerun] submitting {rendered_path} ({len(spec['tasks'])} tasks) to {workspace}")
    out = subprocess.run(
        [BEAKER, "experiment", "create", "-w", workspace, str(rendered_path)],
        capture_output=True, text=True,
    )
    print(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit(f"[heldout-rerun] `beaker experiment create` failed (exit {out.returncode})")
    m = re.search(r"(?:Experiment\s+|/ex/)(\S+?)(?:\s+submitted|\s|$)", out.stdout + out.stderr)
    return m.group(1) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-exp", required=True, help="Source training Beaker experiment id.")
    p.add_argument("--variant", required=True, help="e.g. v1 / v2 (group = heldout-rerun-<variant>@<launch_id>).")
    p.add_argument("--cluster", default="ai1/octo-hub-onprem-h200")
    p.add_argument("--workspace", default="ai1/aind-dynamic-foraging-foundation-model")
    p.add_argument("--limit", type=int, default=None, help="Only the first N source tasks.")
    p.add_argument("--only-ratio", default=None, help="Filter to this subject_ratio.")
    p.add_argument("--only-seed", default=None, help="Filter to this seed.")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument(
        "--wrapper-ref", default=WRAPPER_REF,
        help="Wrapper git ref the in-container entrypoint checks out (default the "
             "per_subject_likelihood commit). Override to a newer commit when the "
             "config relies on a feature added after the default pin (e.g. the "
             "few-shot adapt_sessions_per_subject knob).",
    )
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    p.add_argument(
        "--heldout-config", default=CONFIG_REL,
        help="Dispatcher-relative held-out config the in-container finetune points at "
             "(default heldout_offline.yaml; pass the data-scaling-law/heldout_zeroshot.yaml "
             "path for no-adaptation zero-shot — group becomes heldout-zeroshot-<variant>@...).",
    )
    args = p.parse_args()

    all_tasks = enumerate_source_tasks(args.source_exp)
    print(f"[heldout-rerun] source exp {args.source_exp}: {len(all_tasks)} tasks")
    tasks = all_tasks
    if args.only_ratio is not None:
        tasks = [t for t in tasks if abs(float(t["subject_ratio"]) - float(args.only_ratio)) < 1e-9]
    if args.only_seed is not None:
        tasks = [t for t in tasks if t["seed"] == str(args.only_seed)]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        sys.exit("[heldout-rerun] no source tasks matched the filters")
    print(f"[heldout-rerun] selected {len(tasks)} task(s):")
    for t in tasks:
        print(f"    D={t['subject_ratio']} seed={t['seed']} dataset={t['result_dataset']}")

    launch_id = _seattle_launch_id()
    spec, group = build_spec(args.source_exp, args.variant, tasks, args.cluster,
                             launch_id, args.heldout_config, args.wrapper_ref)
    # File-name stem mirrors the W&B group prefix so zero-shot records don't clobber
    # the adapted-rerun records in the same output dir.
    record_stem = group.split("@")[0]  # e.g. heldout-rerun-v1 / heldout-zeroshot-v1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = output_dir / f"{record_stem}_{launch_id}.yaml"

    experiment_id = None
    if args.no_submit:
        rendered.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(f"[heldout-rerun] --no-submit; wrote {rendered}")
    else:
        experiment_id = submit(spec, args.workspace, rendered)

    record = {
        "kind": record_stem.rsplit("-", 1)[0],  # heldout-rerun / heldout-zeroshot
        "heldout_config": args.heldout_config,
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
    print(f"[heldout-rerun] saved record: {rec_path}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
