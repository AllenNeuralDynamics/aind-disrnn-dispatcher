#!/usr/bin/env python
"""Diagnostic: dump per-subject fitted-vs-true params for the 4 baseline-rl stage-2 runs.
Writes stage2_baseline_persubject.csv (one row per subject per run) so we can see WHY
softmax_temp recovery R2 is negative: drift-driven scatter, a few blown-up outliers, or
a scale mismatch. Reuses the SDK-free fetch + nested-config GT regen from the recompute.
Run on an HPC compute node (disrnn-cpu, WANDB_API_KEY exported)."""
import os, sys, json, io
import numpy as np, pandas as pd, requests
from sklearn.metrics import r2_score

ENT, PROJ = "AIND-disRNN", "embedding_recovery"
RUNS = {50: "f787vi1g", 100: "uooa8w1k", 200: "0l8d5rq9", 300: "col8bdj9"}
GQL = "https://api.wandb.ai/graphql"; AUTH = ("api", os.environ["WANDB_API_KEY"])
GT_COL = {"biasL": "param_biasL", "learn_rate": "param_learn_rate",
          "softmax_temp": "param_softmax_inverse_temperature"}
FIT_COL = {"biasL": "biasL", "learn_rate": "learn_rate",
           "softmax_temp": "softmax_inverse_temperature"}


def _gql(q, v):
    r = requests.post(GQL, json={"query": q, "variables": v}, auth=AUTH, timeout=90)
    r.raise_for_status(); j = r.json()
    if j.get("errors"): raise RuntimeError(j["errors"])
    return j["data"]


def table_url(run):
    q = """query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){
      run(name:$r){files(first:200){edges{node{name directUrl}}}}}}"""
    for e in _gql(q, {"e": ENT, "p": PROJ, "r": run})["project"]["run"]["files"]["edges"]:
        if "subject_fit_metrics" in e["node"]["name"] and e["node"]["name"].endswith(".table.json"):
            return e["node"]["directUrl"]
    raise FileNotFoundError(run)


def run_cfg(run):
    q = """query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){config}}}"""
    return json.loads(_gql(q, {"e": ENT, "p": PROJ, "r": run})["project"]["run"]["config"])


def regen_gt(cfg):
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
    return ld.groundtruth_table().groupby("subject_id")[list(GT_COL.values())].mean().reset_index()


if __name__ == "__main__":
    allrows = []
    for N, rid in RUNS.items():
        j = requests.get(table_url(rid), timeout=120).json()
        fit = pd.DataFrame(j["data"], columns=j["columns"])
        if fit["subject_index"].dtype != object:
            fit["subject_id"] = fit["subject_index"].map(lambda i: f"synth{int(i):03d}")
        gt = regen_gt(run_cfg(rid))
        m = fit.merge(gt, on="subject_id", how="inner")
        for p in GT_COL:
            t, f = m[GT_COL[p]].values, m[FIT_COL[p]].values
            print(f"N={N} {p}: R2={r2_score(t,f):.3f}  true[min,med,max]=[{t.min():.2f},{np.median(t):.2f},{t.max():.2f}]  "
                  f"fit[min,med,max]=[{f.min():.2f},{np.median(f):.2f},{f.max():.2f}]  "
                  f"n_fit_gt10={int((f>10).sum())} n_fit_at15={int((np.abs(f-15)<0.01).sum())}")
        sub = m[["subject_id"] + list(GT_COL.values()) + list(FIT_COL.values())].copy()
        sub.insert(0, "num_subjects", N); sub.insert(1, "wandb_run_id", rid)
        allrows.append(sub)
    out = pd.concat(allrows, ignore_index=True)
    out.to_csv("stage2_baseline_persubject.csv", index=False)
    print("WROTE stage2_baseline_persubject.csv", out.shape)
