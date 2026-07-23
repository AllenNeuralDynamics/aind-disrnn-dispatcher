"""Pull the mult-d-grid W&B group into a committed CSV.

The CSV is the source of truth for the report producer (``scaling_report.py``), so ``make``
runs offline; re-run this only to refresh from W&B (needs WANDB_API_KEY). Safe to run while
the grid is still in flight — includes runs in every state (running/crashed/failed/finished);
``scaling_report.py`` is responsible for filtering to ``state == "finished"`` before trusting
any metric (heldout/eval_likelihood is written incrementally throughout training, so a
not-yet-finished run's value is not final).

    python analysis/pull_grid.py

Writes analysis/grid.csv — one row per run: D, mult, beta, seed, state, heldout_ll, eval_ll.
"""
from __future__ import annotations

import csv
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
PROJECT = "AIND-disRNN/disrnn_data_scaling"
GROUP = "mult-d-grid@20260718-151409"


def main() -> None:
    api = wandb.Api()
    rows = []
    for r in api.runs(PROJECT, filters={"group": GROUP}, per_page=200):
        cfg, s = r.config, r.summary
        model = cfg.get("model") or {}
        pen = model.get("penalties") or {}
        beta = pen.get("beta")
        upl = pen.get("update_net_latent_penalty")
        ids = cfg.get("resolved_subject_ids") or []
        row = {
            "run_id": r.id,
            "wandb_run_name": r.name,
            "state": r.state,
            "D": len(ids) or None,           # ACTUAL resolved D, never the nominal ratio
            "seed": (cfg.get("data") or {}).get("seed"),
            "beta": beta,
            # effective multiplier is recovered post-hoc: the dispatcher consumes and drops
            # the multiplier field before training (same trick as study 05's pull_grid.py)
            "mult": round(upl / beta) if (beta and upl) else None,
            "heldout_ll": s.get("heldout/eval_likelihood"),
            "eval_ll": s.get("checkpoint/eval_likelihood"),
            "final_step": s.get("_step"),
        }
        rows.append(row)

    rows.sort(key=lambda x: (x["D"] or 0, x["mult"] or 0, x["beta"] or 0, x["seed"] or 0))
    out = HERE / "grid.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    done = sum(1 for r in rows if r["state"] == "finished")
    print(f"wrote {out} — {len(rows)} runs seen ({done} finished, {len(rows)-done} in flight/failed)")


if __name__ == "__main__":
    main()
