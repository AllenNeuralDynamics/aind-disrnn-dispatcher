"""Poll a study's Beaker experiments + W&B runs and append timestamped snapshots.

WHY THIS EXISTS
---------------
An agent session only executes while a turn is running: it cannot wake itself on a
timer, so "report every N hours" cannot be delivered by the agent alone. What it
CAN do is make sure no history is lost between check-ins. Current state is always
queryable, but *transitions* are not: a task that was evicted and auto-resumed, a
run that crashed and was retried, the moment a cell crossed into held-out scoring
— all of that is invisible by the time someone next looks.

This watcher runs unattended, appends one JSON line per poll, and derives an
event log from consecutive snapshots. On the next check-in the answer to "what
happened while I was away" is a file read, not a guess.

USAGE
    python studies/util/watch_runs.py \
        --experiments <label>=<beaker_id> [<label>=<beaker_id> ...] \
        --wandb-project mice_rt_lick_scaling \
        --group-prefix h128-dscan \
        --out-dir /tmp/study07_watch \
        --interval-s 600 --max-hours 12

Reads BEAKER_TOKEN and WANDB_API_KEY from the environment. Writes:
    snapshots.jsonl   one record per poll (full state)
    events.jsonl      derived transitions only
    status.md         human-readable latest status, overwritten each poll

Exits when every tracked experiment is terminal, or at --max-hours.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger("watch_runs")

PT = ZoneInfo("America/Los_Angeles")
LIVE_STATES = {"running", "idle", "scheduled", "created", "initializing"}

_WANDB_QUERY = """query R($e:String!,$p:String!,$c:String){
  project(name:$p,entityName:$e){
    runs(first:200, after:$c){
      pageInfo{hasNextPage endCursor}
      edges{node{name group state config summaryMetrics}}}}}"""


def _wandb_runs(entity: str, project: str, key: str):
    """Page through every run in a project (the API caps a page at 200)."""
    cursor = None
    while True:
        body = json.dumps(
            {"query": _WANDB_QUERY, "variables": {"e": entity, "p": project, "c": cursor}}
        ).encode()
        auth = base64.b64encode(f"api:{key}".encode()).decode()
        req = urllib.request.Request(
            "https://api.wandb.ai/graphql",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
        )
        payload = json.load(urllib.request.urlopen(req, timeout=60))
        page = payload["data"]["project"]["runs"]
        for edge in page["edges"]:
            yield edge["node"]
        if not page["pageInfo"]["hasNextPage"]:
            return
        cursor = page["pageInfo"]["endCursor"]


def _unwrap(cfg, *path):
    """Walk a W&B config, transparently unwrapping the {'value': ...} boxes."""
    cur = cfg
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if isinstance(cur, dict) and "value" in cur and key != "value":
            cur = cur["value"]
    return cur


def poll(beaker, experiments: dict[str, str], entity: str, project: str,
         group_prefix: str, key: str) -> dict:
    """One snapshot: Beaker task states plus W&B per-run progress."""
    snap = {"ts": datetime.now(timezone.utc).isoformat(), "experiments": {}, "runs": {}}

    for label, exp_id in experiments.items():
        try:
            exp = beaker.experiment.get(exp_id)
            jobs = [j for j in (exp.jobs or []) if str(j.status.current) != "canceled"]
            states = Counter(str(j.status.current) for j in jobs)
            snap["experiments"][label] = {
                "id": exp_id,
                "states": dict(states),
                "live": sum(v for k, v in states.items() if k in LIVE_STATES),
                # status.message is where a pull failure or eviction reason shows up;
                # exit_code alone does not reveal either.
                "messages": sorted(
                    {
                        (getattr(j.status, "message", None) or "")[:160]
                        for j in jobs
                        if getattr(j.status, "message", None)
                    }
                ),
            }
        except Exception as exc:  # a transient API error must not kill the watcher
            snap["experiments"][label] = {"id": exp_id, "error": f"{type(exc).__name__}: {exc}"}

    try:
        for node in _wandb_runs(entity, project, key):
            group = (node.get("group") or "").split("@")[0]
            if group_prefix and not group.startswith(group_prefix):
                continue
            cfg = json.loads(node["config"] or "{}")
            summary = json.loads(node["summaryMetrics"] or "{}")
            timing = _unwrap(cfg, "data", "timing_features") or {}
            snap["runs"][node["name"]] = {
                "group": group,
                "state": node["state"],
                "arm": "ON" if timing.get("enabled") else "OFF",
                "n_subjects": len(_unwrap(cfg, "resolved_subject_ids") or []),
                "subject_ratio": _unwrap(cfg, "data", "subject_ratio"),
                "step": summary.get("_step"),
                "runtime_s": summary.get("_runtime"),
                "heldout": (
                    summary.get("heldout/final/eval_likelihood")
                    or summary.get("heldout/eval_likelihood")
                ),
                "within": summary.get("checkpoint/eval_likelihood"),
            }
    except Exception as exc:
        snap["wandb_error"] = f"{type(exc).__name__}: {exc}"
    return snap


def diff_events(prev: dict | None, cur: dict) -> list[dict]:
    """Transitions worth recording. Empty list when nothing changed."""
    if prev is None:
        return [{"ts": cur["ts"], "kind": "watch_start",
                 "detail": f"tracking {len(cur['experiments'])} experiments, "
                           f"{len(cur['runs'])} runs"}]
    events = []
    for label, now in cur["experiments"].items():
        was = prev["experiments"].get(label, {})
        if was.get("states") != now.get("states"):
            events.append({"ts": cur["ts"], "kind": "beaker_states", "target": label,
                           "detail": f"{was.get('states')} -> {now.get('states')}"})
        new_msgs = set(now.get("messages") or []) - set(was.get("messages") or [])
        for msg in sorted(new_msgs):
            events.append({"ts": cur["ts"], "kind": "beaker_message",
                           "target": label, "detail": msg})
    for name, now in cur["runs"].items():
        was = prev["runs"].get(name)
        if was is None:
            events.append({"ts": cur["ts"], "kind": "run_appeared", "target": name,
                           "detail": f"{now['group']} arm={now['arm']} D={now['n_subjects']}"})
            continue
        if was["state"] != now["state"]:
            events.append({"ts": cur["ts"], "kind": "run_state", "target": name,
                           "detail": f"{was['state']} -> {now['state']} at step {now['step']}"})
        if was.get("heldout") is None and now.get("heldout") is not None:
            events.append({"ts": cur["ts"], "kind": "heldout_landed", "target": name,
                           "detail": f"arm={now['arm']} D={now['n_subjects']} "
                                     f"heldout={now['heldout']:.5f}"})
        # A step going BACKWARDS is the fingerprint of a preemption + auto-resume:
        # the task restarts from its last checkpoint, so this is the only reliable
        # in-band signal that an eviction happened at all.
        if (was.get("step") or 0) - (now.get("step") or 0) > 500:
            events.append({"ts": cur["ts"], "kind": "step_regressed", "target": name,
                           "detail": f"step {was['step']} -> {now['step']} "
                                     f"(preemption + resume?)"})
    return events


def render(cur: dict, events: list[dict], out_dir: Path) -> str:
    """Overwrite status.md with the latest state, newest events last."""
    now_pt = datetime.now(PT)
    lines = [f"# Watch status — {now_pt:%Y-%m-%d %H:%M} PT", ""]
    for label, exp in cur["experiments"].items():
        if "error" in exp:
            lines.append(f"- **{label}**: query error — {exp['error']}")
            continue
        lines.append(f"- **{label}**: {exp['states']} (live={exp['live']})")
        for msg in exp.get("messages") or []:
            lines.append(f"    - msg: `{msg}`")
    runs = cur["runs"]
    scored = {k: v for k, v in runs.items() if v.get("heldout") is not None}
    lines += ["", f"Runs tracked: {len(runs)} · with held-out: {len(scored)}", ""]

    by_cell: dict[tuple, list] = {}
    for v in runs.values():
        by_cell.setdefault((v["arm"], v["subject_ratio"]), []).append(v)
    lines += ["| arm | ratio | n | scored | mean held-out | max step |", "|---|---|---|---|---|---|"]
    for (arm, ratio), vs in sorted(by_cell.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        hs = [v["heldout"] for v in vs if v.get("heldout") is not None]
        mean = f"{sum(hs)/len(hs):.5f}" if hs else "—"
        steps = [v["step"] or 0 for v in vs]
        lines.append(f"| {arm} | {ratio} | {len(vs)} | {len(hs)} | {mean} | {max(steps)} |")

    if events:
        lines += ["", "## New events this poll"]
        lines += [f"- `{e['kind']}` {e.get('target','')} — {e['detail']}" for e in events]
    text = "\n".join(lines) + "\n"
    (out_dir / "status.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True,
                    help="label=beaker_experiment_id pairs")
    ap.add_argument("--wandb-entity", default="AIND-disRNN")
    ap.add_argument("--wandb-project", required=True)
    ap.add_argument("--group-prefix", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval-s", type=int, default=600)
    ap.add_argument("--max-hours", type=float, default=12.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[watch] %(asctime)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = dict(pair.split("=", 1) for pair in args.experiments)
    key = os.environ["WANDB_API_KEY"]

    from beaker import Beaker, Config
    beaker = Beaker(
        Config(user_token=os.environ["BEAKER_TOKEN"], default_org="ai1",
               default_workspace="ai1/aind-dynamic-foraging-foundation-model")
    )

    deadline = datetime.now(timezone.utc) + timedelta(hours=args.max_hours)
    prev = None
    while datetime.now(timezone.utc) < deadline:
        cur = poll(beaker, experiments, args.wandb_entity, args.wandb_project,
                   args.group_prefix, key)
        events = diff_events(prev, cur)
        with (out_dir / "snapshots.jsonl").open("a") as fh:
            fh.write(json.dumps(cur) + "\n")
        if events:
            with (out_dir / "events.jsonl").open("a") as fh:
                for event in events:
                    fh.write(json.dumps(event) + "\n")
        render(cur, events, out_dir)
        logger.info(
            "poll ok — %d runs, %d scored, %d new events",
            len(cur["runs"]),
            sum(1 for v in cur["runs"].values() if v.get("heldout") is not None),
            len(events),
        )
        prev = cur

        live = sum(e.get("live", 0) for e in cur["experiments"].values()
                   if isinstance(e, dict))
        if live == 0:
            logger.info("all tracked experiments terminal — exiting")
            with (out_dir / "events.jsonl").open("a") as fh:
                fh.write(json.dumps({"ts": cur["ts"], "kind": "watch_done",
                                     "detail": "no live tasks remain"}) + "\n")
            return
        time.sleep(args.interval_s)
    logger.info("max-hours reached — exiting")


if __name__ == "__main__":
    main()
