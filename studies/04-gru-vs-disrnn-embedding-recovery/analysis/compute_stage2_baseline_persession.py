#!/usr/bin/env python
"""Baseline_rl PER-SESSION parameter recovery -> stage2_baseline_persession_recovery.csv.

Companion to compute_stage2_baseline_recovery.py (which scores against the session-MEAN).
Here the target is each SESSION's true drifting parameter: the fixed per-subject fit is
broadcast to all of that subject's sessions and scored r2 against the per-session GT.
This is the baseline analog of the GRU 'subject-only' (session-blind) per-session recovery
in stage2_session_traj.py — a fixed per-subject value, no within-subject session info.

Run on an HPC compute node (disrnn-cpu, WANDB_API_KEY exported; wrapper importable for
per-session GT regen). SDK-free: GraphQL for file URLs + config, requests.get for GCS.

softmax_inverse_temperature is winsorized at 20 (true ceiling ~18.6) for the same reason
as the session-mean recompute: 5-10 near-deterministic subjects have divergent per-subject
beta MLEs. Raw R2, Spearman, n_winsorized kept as disclosed columns.

spearman_biasL / spearman_learn_rate report the Spearman rank correlation for those two
params alongside their R^2 (same y_true/y_pred pair; R^2 stays the primary metric).
"""
import os, sys, json, io
import numpy as np, pandas as pd, requests
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

ENT, PROJ = "AIND-disRNN", "embedding_recovery"
RUNS = {50: "f787vi1g", 100: "uooa8w1k", 200: "0l8d5rq9", 300: "col8bdj9"}
GQL = "https://api.wandb.ai/graphql"; AUTH = ("api", os.environ["WANDB_API_KEY"])
GT_COL = {"biasL": "param_biasL", "learn_rate": "param_learn_rate",
          "softmax_temp": "param_softmax_inverse_temperature"}
FIT_COL = {"biasL": "biasL", "learn_rate": "learn_rate",
           "softmax_temp": "softmax_inverse_temperature"}
WINSOR = 20.0


def _gql(q, v):
    r = requests.post(GQL, json={"query": q, "variables": v}, auth=AUTH, timeout=90)
    r.raise_for_status(); j = r.json()
    if j.get("errors"): raise RuntimeError(j["errors"])
    return j["data"]


def table_url(run):
    q = """query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){
      run(name:$r){files(first:200){edges{node{name directUrl}}}}}}"""
    for e in _gql(q, {"e": ENT, "p": PROJ, "r": run})["project"]["run"]["files"]["edges"]:
        n = e["node"]["name"]
        if "subject_fit_metrics" in n and n.endswith(".table.json"):
            return e["node"]["directUrl"]
    raise FileNotFoundError(run)


def run_cfg(run):
    q = """query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){config}}}"""
    return json.loads(_gql(q, {"e": ENT, "p": PROJ, "r": run})["project"]["run"]["config"])


def regen_gt_persession(cfg):
    """FULL per-session GT table (one row per subject-session), NOT grouped."""
    sys.path.insert(0, os.path.expanduser("~/code/aind-disrnn-wrapper/code"))
    os.environ.setdefault("DISRNN_GEN_WORKERS", "1")
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
    d = cfg["data"]["value"] if isinstance(cfg["data"], dict) and "value" in cfg["data"] else cfg["data"]
    ld = HierarchicalCognitiveAgents(
        task=d["task"], agent=d["agent"], num_trials=d["num_trials"],
        num_subjects=d["num_subjects"], num_sessions_per_subject=d["num_sessions_per_subject"],
        eval_every_n=d.get("eval_every_n", 2), batch_size=d.get("batch_size"),
        batch_mode=d.get("batch_mode", "random"),
        subject_seed_stride=d.get("subject_seed_stride", 100000),
        generation_workers=1, seed=d.get("seed", 42))
    return ld.groundtruth_table()  # per-session rows


if __name__ == "__main__":
    rows = []
    for N, rid in RUNS.items():
        j = requests.get(table_url(rid), timeout=120).json()
        fit = pd.DataFrame(j["data"], columns=j["columns"])
        if fit["subject_index"].dtype != object:
            fit["subject_id"] = fit["subject_index"].map(lambda i: f"synth{int(i):03d}")
        gt = regen_gt_persession(run_cfg(rid))  # subject_id, param_*, per session
        # broadcast the fixed per-subject fit onto every session of that subject
        keep_fit = ["subject_id"] + list(FIT_COL.values())
        m = gt.merge(fit[keep_fit], on="subject_id", how="inner")
        n_sess = len(m) // N
        assert len(m) % N == 0, f"rows {len(m)} not divisible by N={N}"
        row = {"num_subjects": N, "wandb_run_id": rid, "n_session_rows": len(m)}
        r2s = []
        for p in GT_COL:
            t = m[GT_COL[p]].values; f = m[FIT_COL[p]].values
            if p == "softmax_temp":
                fw = np.clip(f, None, WINSOR)
                row["r2_softmax_temp"] = r2_score(t, fw)
                row["r2_softmax_temp_raw"] = r2_score(t, f)
                row["softmax_spearman"] = pd.Series(f).corr(pd.Series(t), method="spearman")
                row["n_softmax_winsorized_rows"] = int((f > WINSOR).sum())
                r2s.append(row["r2_softmax_temp"])
            else:
                row[f"r2_{p}"] = r2_score(t, f)
                rho, _pval = spearmanr(t, f)
                row[f"spearman_{p}"] = float(rho)
                r2s.append(row[f"r2_{p}"])
        row["r2_mean"] = float(np.mean(r2s))
        row["softmax_winsor_threshold"] = WINSOR
        rows.append(row)
        print(f"N={N} {rid}: per-session biasL={row['r2_biasL']:.3f} learn={row['r2_learn_rate']:.3f} "
              f"softmax={row['r2_softmax_temp']:.3f} mean={row['r2_mean']:.3f} ({n_sess} sess/subj) "
              f"spearman_biasL={row['spearman_biasL']:.3f} spearman_learn_rate={row['spearman_learn_rate']:.3f}")
    out = pd.DataFrame(rows)[
        ["num_subjects", "r2_biasL", "spearman_biasL", "r2_learn_rate", "spearman_learn_rate",
         "r2_softmax_temp", "r2_mean", "r2_softmax_temp_raw", "softmax_spearman",
         "n_softmax_winsorized_rows", "softmax_winsor_threshold", "n_session_rows", "wandb_run_id"]]
    out.to_csv("stage2_baseline_persession_recovery.csv", index=False)
    print("WROTE stage2_baseline_persession_recovery.csv", out.shape)
