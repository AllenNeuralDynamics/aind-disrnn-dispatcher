"""Pull the 05-disrnn-scaling-law grid from W&B into a committed CSV.

The CSV is the source of truth for the report producer (``scaling_report.py``), so
``make`` runs offline; re-run this only to refresh from W&B (needs WANDB_API_KEY).

    python analysis/pull_grid.py

Writes analysis/grid.csv — one row per run, with the two y-axes this study cares about:
  * heldout/eval_likelihood        — held-out-mouse transfer (the scaling-law y-axis)
  * final/bottlenecks/<fam>_total_openness = Sigma(1-sigma) for the six bottleneck families

NEVER use n_eff_open_frac: it is scale-invariant and reports a spuriously high value even
for a fully shut bottleneck (study 03 metric caveat).
"""
from __future__ import annotations

import csv
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
PROJECT = "AIND-disRNN/disrnn_data_scaling"
GROUPS = [
    "dscan-mult2@20260713-003428",       # wave 1 — the scaling curve
    "mult-beta-d614@20260713-003501",    # wave 2 — mult x beta at the full cohort
    "subject-capacity@20260713-225831",  # wave 3 — subject capacity (may be incomplete)
]
FAMILIES = [
    "update_net_latent",   # the INTERACTION bottleneck the multiplier targets
    "update_net_obs",
    "update_net_subj",
    "latent",              # recurrent
    "choice_net_latent",
    "choice_net_subj",
]


def main() -> None:
    api = wandb.Api()
    rows = []
    for group in GROUPS:
        for r in api.runs(PROJECT, filters={"group": group}):
            cfg, s = r.config, r.summary
            model = cfg.get("model") or {}
            arch = model.get("architecture") or {}
            pen = model.get("penalties") or {}
            beta = pen.get("beta")
            upl = pen.get("update_net_latent_penalty")
            ids = cfg.get("resolved_subject_ids") or []
            row = {
                "group": group,
                "variant": group.split("@")[0],
                "run_id": r.id,
                "state": r.state,
                "D": len(ids),                       # ACTUAL resolved D, never the nominal ratio
                "seed": cfg.get("seed"),
                "beta": beta,
                # effective multiplier is recovered post-hoc: the dispatcher consumes and drops
                # the multiplier field before training
                "mult": round(upl / beta) if (beta and upl) else None,
                "subject_embedding_size": arch.get("subject_embedding_size"),
                "subject_penalty": pen.get("subject_penalty"),
                "heldout_ll": s.get("heldout/eval_likelihood"),
                "eval_ll": s.get("checkpoint/eval_likelihood"),
            }
            for fam in FAMILIES:
                row[f"open_{fam}"] = s.get(f"final/bottlenecks/{fam}_total_openness")
            rows.append(row)

    rows.sort(key=lambda x: (x["variant"], x["D"] or 0, x["mult"] or 0, x["seed"] or 0))
    out = HERE / "grid.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    done = sum(1 for r in rows if r["state"] == "finished")
    print(f"wrote {out} — {len(rows)} runs ({done} finished)")


if __name__ == "__main__":
    main()
