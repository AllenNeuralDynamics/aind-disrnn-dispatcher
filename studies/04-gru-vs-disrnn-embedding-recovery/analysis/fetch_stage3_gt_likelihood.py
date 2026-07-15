#!/usr/bin/env python
"""Fetch groundtruth_likelihood (W&B summary metric, logged per-run) from the finished
stage-3 GRU sweep runs -- needed to convert baseline's raw eval_likelihood into the same
likelihood_relative_to_groundtruth metric GRU already reports, for a fair fit-quality
comparison. SDK-free GraphQL (same pattern as stage1_gt_likelihood.json)."""
import os, json, requests, subprocess

key = subprocess.run(
    ["bash", "-lc", "grep -A2 api.wandb.ai ~/.netrc | grep password | awk '{print $2}'"],
    capture_output=True, text=True).stdout.strip()

Q = """query($e:String!,$p:String!,$s:String!){project(name:$p,entityName:$e){
  sweep(sweepName:$s){runs(first:20){edges{node{name state summaryMetrics}}}}}}"""

r = requests.post("https://api.wandb.ai/graphql",
                   json={"query": Q, "variables": {"e": "AIND-disRNN", "p": "embedding_recovery", "s": "ychlajgl"}},
                   auth=("api", key), timeout=30).json()
if "errors" in r and r["errors"]:
    raise RuntimeError(r["errors"])
edges = r["data"]["project"]["sweep"]["runs"]["edges"]

rows = []
for e in edges:
    n = e["node"]
    if n["state"] != "finished":
        continue
    sm = json.loads(n["summaryMetrics"])
    rows.append({"run": n["name"], "lik_rel": sm.get("likelihood_relative_to_groundtruth"),
                 "groundtruth_likelihood": sm.get("groundtruth_likelihood")})
    print(n["name"], rows[-1]["lik_rel"], rows[-1]["groundtruth_likelihood"])

import pandas as pd
pd.DataFrame(rows).to_csv("stage3_gru_gt_likelihood.csv", index=False)
print("WROTE stage3_gru_gt_likelihood.csv")
