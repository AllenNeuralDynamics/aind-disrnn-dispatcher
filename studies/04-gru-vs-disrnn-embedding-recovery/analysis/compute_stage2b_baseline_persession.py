#!/usr/bin/env python
"""Stage-2b baseline_rl PER-SESSION parameter recovery -> stage2b_baseline_persession.csv.

Same method as compute_stage2_baseline_persession.py (broadcast the fixed per-subject fit
to every session, score r2 against the per-session GT over ALL sessions — matching how the
GRU trajectory-recovery numbers in stage2b_session_trajectory_recovery.csv were computed),
applied to the single baseline-rl-stage2b run (N=200, sweep ykjk89o5).

Run on an HPC compute node (disrnn-cpu, WANDB_API_KEY exported; wrapper importable).
SDK-free: GraphQL for run discovery/config/file URLs, requests.get for GCS.

softmax_inverse_temperature winsorized at 20 (true ceiling ~18.6) for the same reason as
the stage-2 recompute. Raw R2, Spearman, n_winsorized kept as disclosed columns.
"""
import os, sys, json
import numpy as np, pandas as pd, requests
from sklearn.metrics import r2_score

ENT, PROJ = "AIND-disRNN", "embedding_recovery"
SWEEP = "ykjk89o5"
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


def find_run():
    q = """query($e:String!,$p:String!,$s:String!){project(name:$p,entityName:$e){
      sweep(sweepName:$s){runs(first:20){edges{node{name state}}}}}}"""
    edges = _gql(q, {"e": ENT, "p": PROJ, "s": SWEEP})["project"]["sweep"]["runs"]["edges"]
    finished = [e["node"]["name"] for e in edges if e["node"]["state"] == "finished"]
    assert len(finished) == 1, f"expected 1 finished run in sweep {SWEEP}, got {finished}"
    return finished[0]


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
    return ld.groundtruth_table()  # per-session rows, ALL sessions


if __name__ == "__main__":
    rid = find_run()
    print("baseline-rl-stage2b run:", rid)
    j = requests.get(table_url(rid), timeout=120).json()
    fit = pd.DataFrame(j["data"], columns=j["columns"])
    if fit["subject_index"].dtype != object:
        fit["subject_id"] = fit["subject_index"].map(lambda i: f"synth{int(i):03d}")
    gt = regen_gt_persession(run_cfg(rid))
    keep_fit = ["subject_id"] + list(FIT_COL.values())
    m = gt.merge(fit[keep_fit], on="subject_id", how="inner")
    N = m.subject_id.nunique()
    row = {"stage": "S2b", "cond": "baseline_rl", "num_subjects": N,
           "n_session_rows": len(m), "wandb_run_id": rid}
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
            row[f"r2_{p}"] = r2_score(t, f); r2s.append(row[f"r2_{p}"])
    row["r2_mean"] = float(np.mean(r2s))
    row["softmax_winsor_threshold"] = WINSOR
    print(f"S2b baseline_rl persession: biasL={row['r2_biasL']:.3f} learn={row['r2_learn_rate']:.3f} "
          f"softmax={row['r2_softmax_temp']:.3f} mean={row['r2_mean']:.3f} ({len(m)} session rows)")
    pd.DataFrame([row]).to_csv("stage2b_baseline_persession.csv", index=False)
    print("WROTE stage2b_baseline_persession.csv")
