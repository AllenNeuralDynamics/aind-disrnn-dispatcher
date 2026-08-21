#!/usr/bin/env python3
"""Recover `heldout/eval_likelihood` for runs whose final summary write was lost.

WHY THIS EXISTS. Some mult-d-grid tasks completed training AND their held-out fine-tune
successfully -- Beaker exit 0, logs end with "All done, goodbye", and W&B holds their COMMITTED
`disrnn-output-*` (training-output) and `*-heldoutper_subject_likelihood` (run_table) artifacts.
But each had been preempted several times, W&B marked the run `crashed` on a heartbeat timeout,
and the *final summary write* was silently dropped: the run objects sit frozen at a mid-training
step with NO `heldout/*` key at all. Beaker considers these tasks done (exit 0), so autoResume
never retried them -- the metric was simply lost, and `scaling_report.py`'s
`state == "finished"` filter drops those cells. First hit on 5 tasks (-049/-052/-054/-056/-057,
2026-07-25), then recurred on -058 hours later; expect more as long runs keep getting preempted.
See the beaker-launch skill, `references/scheduling-lessons.md` "exit 0 with a missing metric".

SELF-DISCOVERING. Rather than a hand-maintained ID list (the failure recurs, so a list goes
stale), this scans the W&B group for runs matching ALL of:

  * `state == "crashed"`      -- not finished, not still running, not a real `failed` error
  * NO `heldout/eval_likelihood` key in the summary
  * a COMMITTED `run_table` artifact holding `per_subject_likelihood.table.json`

**The table is the safety interlock.** The held-out stage runs only at the very END of a run, so
its per-subject table cannot exist unless that stage completed. A run that was merely preempted
mid-training has no such table and is therefore never touched -- there is no way to mistake an
in-progress run for a recoverable one.

WHY NOT re-score on GPU. The documented re-score path
(`resume_heldout_beaker.py --run-id <id>`, see the beaker-launch skill's
`references/resume-extend-rescore.md`) re-runs the held-out fine-tune off the checkpoint tree.
That is unnecessary here: the held-out stage ALREADY RAN and its full per-subject output
survived as a committed W&B run_table. The scalar is a pure function of that table, so it can be
recovered EXACTLY -- no GPU, no re-training, no risk of a different result.

THE AGGREGATION, VERIFIED (not assumed). `heldout/eval_likelihood` is the TRIAL-WEIGHTED
GEOMETRIC mean of the per-subject likelihoods (correct, since a likelihood is exp(mean
log-lik/trial)):

    heldout/eval_likelihood = exp( sum_i n_trials_i * ln(lik_i) / sum_i n_trials_i )

Validated against 5 independent runs that have BOTH the table and a natively-logged scalar:
reproduces each to <= 1e-7. The two naive alternatives are badly wrong and were rejected --
simple mean over subjects is off by ~0.005 and the arithmetic trial-weighted mean by ~0.004,
i.e. the same magnitude as the effects this study measures.

WHAT IT WRITES. The recovered value into each run's W&B summary under the standard key, PLUS a
provenance marker so a backfilled value is never mistaken for a natively-logged one:

    heldout/eval_likelihood              <- recovered value
    heldout/eval_likelihood_backfilled   <- True
    heldout/eval_likelihood_backfill_src <- "per_subject_likelihood.table.json (trial-weighted geometric mean)"

Idempotent: re-running recomputes the same value from the same immutable artifact.

    python analysis/backfill_lost_heldout.py --dry-run   # print, write nothing
    python analysis/backfill_lost_heldout.py             # write to W&B
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import tempfile
from pathlib import Path

import wandb

PROJECT = "AIND-disRNN/disrnn_data_scaling"
GROUP = "mult-d-grid@20260718-151409"
KEY = "heldout/eval_likelihood"
SRC_NOTE = "per_subject_likelihood.table.json (trial-weighted geometric mean)"


def heldout_from_table(run) -> tuple[float, int]:
    """Recover the scalar from the run's per-subject held-out table. Returns (value, n_subjects)."""
    tables = [a for a in run.logged_artifacts() if a.type == "run_table"]
    if not tables:
        raise RuntimeError(f"{run.id}: no run_table artifact -- cannot recover")
    tmp = tempfile.mkdtemp()
    try:
        tables[0].download(root=tmp)
        hits = glob.glob(f"{tmp}/**/per_subject_likelihood.table.json", recursive=True)
        if not hits:
            raise RuntimeError(f"{run.id}: run_table has no per_subject_likelihood table")
        d = json.loads(Path(hits[0]).read_text())
        cols, rows = d["columns"], d["data"]
        li, ni = cols.index("eval_likelihood"), cols.index("n_trials")
        num = sum(r[ni] * math.log(r[li]) for r in rows)
        den = sum(r[ni] for r in rows)
        return math.exp(num / den), len(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def has_heldout_table(run) -> bool:
    """True iff the run has a COMMITTED run_table artifact -- proof the held-out stage ran."""
    return any(a.type == "run_table" and a.state == "COMMITTED" for a in run.logged_artifacts())


def discover_candidates(runs) -> list:
    """Runs that completed their held-out stage but lost the final summary write.

    See the module docstring: state=='crashed' + missing scalar + a COMMITTED held-out table.
    The table is the interlock -- it exists only if the held-out stage finished, so a run that
    was merely preempted mid-training can never be selected.
    """
    out = []
    for run in runs:
        if run.state != "crashed" or run.summary.get(KEY) is not None:
            continue
        if has_heldout_table(run):
            out.append(run)
    return out


def validate_formula(api, n: int = 5) -> None:
    """Re-verify the aggregation on runs that have BOTH a table and a native scalar."""
    checked = 0
    worst = 0.0
    for run in api.runs(PROJECT, filters={"group": GROUP}, per_page=200):
        if run.state != "finished":
            continue
        native = run.summary.get(KEY)
        if native is None or run.summary.get(f"{KEY}_backfilled"):
            continue
        try:
            recovered, _ = heldout_from_table(run)
        except RuntimeError:
            continue
        worst = max(worst, abs(native - recovered))
        checked += 1
        if checked >= n:
            break
    if checked == 0:
        raise RuntimeError("could not validate the aggregation on any run -- refusing to backfill")
    if worst > 1e-5:
        raise RuntimeError(f"aggregation mismatch up to {worst:.2e} on {checked} runs -- refusing to backfill")
    print(f"formula validated on {checked} native runs (max |diff| = {worst:.2e})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print recovered values, write nothing")
    args = p.parse_args()

    api = wandb.Api(timeout=30)
    validate_formula(api)

    runs = list(api.runs(PROJECT, filters={"group": GROUP}, per_page=200))
    already = sum(1 for r in runs if r.summary.get(f"{KEY}_backfilled"))
    candidates = discover_candidates(runs)
    print(f"scanned {len(runs)} runs in {GROUP}: {len(candidates)} recoverable "
          f"(previously backfilled: {already})")

    for run in candidates:
        value, n_subj = heldout_from_table(run)
        cfg = run.config
        d = len(cfg.get("resolved_subject_ids") or [])
        pen = (cfg.get("model") or {}).get("penalties") or {}
        beta, upl = pen.get("beta"), pen.get("update_net_latent_penalty")
        mult = round(upl / beta) if (beta and upl) else None
        seed = (cfg.get("data") or {}).get("seed")
        print(f"  {run.id[-8:]}  D={d:3d} mult={mult} beta={beta:g} seed={seed} "
              f"n_subj={n_subj}  ->  {KEY}={value:.6f}")
        if args.dry_run:
            continue
        run.summary[KEY] = value
        run.summary[f"{KEY}_backfilled"] = True
        run.summary[f"{KEY}_backfill_src"] = SRC_NOTE
        run.summary.update()

    if not candidates:
        print("nothing to backfill")
    else:
        print("dry run -- nothing written" if args.dry_run
              else f"backfilled {len(candidates)} run(s) to W&B")


if __name__ == "__main__":
    main()
