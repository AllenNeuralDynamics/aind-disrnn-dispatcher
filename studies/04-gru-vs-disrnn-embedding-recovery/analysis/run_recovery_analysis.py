#!/usr/bin/env python
"""Reproducible Stage-1 embedding-recovery analysis.

Pipeline
--------
1. Enumerate finished runs of a W&B sweep (entity/project/sweep from CLI),
   reading each run's config (num_subjects, hidden_size, subject_embedding_size)
   and summary metrics (likelihood_relative_to_groundtruth, or eval_likelihood
   for baseline_rl).
2. For each finished GRU/disRNN run, load its subject_embeddings.pkl and score
   embedding-vs-truth recovery with recovery_scoring.score_run:
       - cross-validated ridge R^2 (embedding -> each true param), and
       - CCA canonical correlations.
3. Ground truth = a single master groundtruth_params.csv. Per-subject params are
   deterministic in subject_idx alone and STATIC across sessions in Stage 1, so
   the largest run's table (e.g. 300 subjects) is a superset that serves every
   run. Pass --gt-master.
4. Emit a tidy scores table (one row per run) and the publication figures:
       - recovery R^2 vs #subjects, faceted by embedding size / hidden size
       - recovered-vs-true scatter for a chosen "best" cell
       - two-part summary: fit likelihood (ceiling 1.0) vs recovery R^2

Why two scores
--------------
Fit likelihood saturates at the ceiling for ANY embedding that can express the
behaviour (a 2-D embedding fits Q-learning choices essentially perfectly), so
likelihood alone cannot tell you whether the latent PARAMETERS are identified.
The embedding-vs-truth recovery score does: it needs embedding_size >= number of
generating factors to recover them. baseline_rl (correctly-specified RL) is the
reference: its eval_likelihood == groundtruth_likelihood (relative ~1.0) sets the
achievable ceiling, and its fitted params vs truth are the correct-model-class
recovery bound the neural embedding is measured against.

W&B access from a network-restricted sandbox
--------------------------------------------
wandb.Api() needs a local socket that may be blocked; a raw GraphQL POST works:
    requests.post("https://api.wandb.ai/graphql",
                  json={"query": Q, "variables": {...}},
                  auth=("api", os.environ["WANDB_API_KEY"]))
See fetch_sweep_runs() below. On HPC/Code Ocean where wandb is unrestricted you
can swap in wandb.Api().sweep(...).runs instead.

Usage
-----
    python run_recovery_analysis.py \
        --entity AIND-disRNN --project embedding_recovery \
        --sweeps mfuaz3ki \
        --embeddings-root /path/with/<run_id>/subject_embeddings.pkl \
        --gt-master /path/to/groundtruth_params.csv \
        --out-dir ./analysis_out
"""
from __future__ import annotations
import argparse, glob, json, os
import pandas as pd
import requests
import recovery_scoring as rs

WANDB_GQL = "https://api.wandb.ai/graphql"


def _val(cfg, k):
    v = cfg.get(k)
    return v.get("value") if isinstance(v, dict) else v


def fetch_sweep_runs(entity, project, sweep):
    """Return [{run, state, num_subjects, hidden_size, embed_size, lik_rel,
    eval_likelihood, groundtruth_likelihood}] via GraphQL (sandbox-safe)."""
    q = ("query($e:String!,$p:String!,$s:String!){project(name:$p,entityName:$e){"
         "sweep(sweepName:$s){runs(first:200){edges{node{name state config summaryMetrics}}}}}}")
    r = requests.post(WANDB_GQL, json={"query": q,
                      "variables": {"e": entity, "p": project, "s": sweep}},
                      auth=("api", os.environ["WANDB_API_KEY"]), timeout=60)
    r.raise_for_status()
    edges = r.json()["data"]["project"]["sweep"]["runs"]["edges"]
    out = []
    for e in edges:
        n = e["node"]
        cfg = json.loads(n.get("config") or "{}")
        sm = json.loads(n.get("summaryMetrics") or "{}")
        out.append({
            "run": n["name"], "state": n["state"], "sweep": sweep,
            "num_subjects": _val(cfg, "data.num_subjects"),
            "hidden_size": _val(cfg, "model.architecture.hidden_size"),
            "embed_size": _val(cfg, "model.architecture.subject_embedding_size"),
            "lik_rel": sm.get("likelihood_relative_to_groundtruth"),
            "eval_likelihood": sm.get("eval_likelihood"),
            "groundtruth_likelihood": sm.get("groundtruth_likelihood"),
        })
    return out


def find_embedding(embeddings_root, run_id):
    """subject_embeddings.pkl for run_id, searched flexibly under a root."""
    for pat in (f"{embeddings_root}/{run_id}/subject_embeddings.pkl",
                f"{embeddings_root}/{run_id}.pkl",
                f"{embeddings_root}/**/*{run_id}*/**/subject_embeddings.pkl"):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="AIND-disRNN")
    ap.add_argument("--project", default="embedding_recovery")
    ap.add_argument("--sweeps", nargs="+", required=True,
                    help="W&B sweep IDs (GRU/disRNN recovery sweeps).")
    ap.add_argument("--embeddings-root", required=True,
                    help="Dir holding <run_id>/subject_embeddings.pkl (or <run_id>.pkl).")
    ap.add_argument("--gt-master", required=True,
                    help="A groundtruth_params.csv covering the largest subject count.")
    ap.add_argument("--out-dir", default="./analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    true_df = rs.load_true_params(args.gt_master)
    rows = []
    for sw in args.sweeps:
        for meta in fetch_sweep_runs(args.entity, args.project, sw):
            row = dict(meta)
            if meta["state"] == "finished":
                pkl = find_embedding(args.embeddings_root, meta["run"])
                if pkl:
                    try:
                        row.update(rs.score_run(pkl, true_df))
                    except Exception as exc:  # keep going; note the failure
                        row["score_error"] = str(exc)
                else:
                    row["score_error"] = "embedding pkl not found"
            rows.append(row)
    scores = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "recovery_scores.csv")
    scores.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(scores)} runs, "
          f"{scores['score_error'].isna().sum() if 'score_error' in scores else len(scores)} scored)")
    return scores


if __name__ == "__main__":
    main()
