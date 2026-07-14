#!/usr/bin/env python
"""Compute stage-2 baseline_rl parameter-recovery R2 -> stage2_baseline_recovery.csv.

WHY THIS RUNS OFF THE CLAUDE-SCIENCE SANDBOX
--------------------------------------------
baseline_rl logs its per-subject fitted parameters as a W&B *media table*
(subject_fit_metrics), whose bytes live on storage.googleapis.com. That host is on
the Claude Science sandbox's exfiltration denylist, and wandb.Api() there fails
(wandb-core needs a socket bind, which the sandbox forbids). Both work fine in a
normal environment: run this on Han's MacBook (VPN + `wandb login` done) OR on the
Allen HPC login node (`disrnn-cpu` env, WANDB_API_KEY exported). Then commit the CSV.

METHOD (baseline recovery = DIRECT fitted-vs-true, NOT embedding regression)
----------------------------------------------------------------------------
baseline_rl fits an independent, correctly-specified ForagerQLearning per subject,
so its recovered params map 1:1 to the 3 generating params. Recovery R2 per param =
r2_score(true_subject_mean, fitted), matching stage-1's stage1_baseline_recovery.csv.
GRU recovery (embedding->param CCA/Ridge) is a different method and lives in
stage2_recovery.py; do not conflate them.

Canonical runs (baseline-rl-stage2@20260704-223824, all finished):
  N=50 f787vi1g, N=100 uooa8w1k, N=200 0l8d5rq9, N=300 col8bdj9

Output columns (frozen, keyed by wandb_run_id per the 'freeze the numbers' rule):
  num_subjects, r2_biasL, r2_learn_rate, r2_softmax_temp, r2_mean, eval_likelihood,
  wandb_run_id
"""
import os, sys, json, io
import numpy as np, pandas as pd, requests
from sklearn.metrics import r2_score

# SDK-free: wandb.Api() crashes on this cluster's wandb 0.23.1 with a simplejson
# 'Object of type Api is not JSON serializable' error during sweep/run resolution
# (traceback in Sweep.get -> gql_request). We hit the
# GraphQL endpoint directly with requests (auth=('api',KEY)) to get file directUrls,
# then plain requests.get() the GCS-signed URLs. GCS is reachable from the compute node.
ENT, PROJ = "AIND-disRNN", "embedding_recovery"
RUNS = {50: "f787vi1g", 100: "uooa8w1k", 200: "0l8d5rq9", 300: "col8bdj9"}
GQL = "https://api.wandb.ai/graphql"
KEY = os.environ["WANDB_API_KEY"]
AUTH = ("api", KEY)


def _gql(query, variables):
    r = requests.post(GQL, json={"query": query, "variables": variables}, auth=AUTH, timeout=90)
    r.raise_for_status()
    j = r.json()
    if "errors" in j and j["errors"]:
        raise RuntimeError(j["errors"])
    return j["data"]


def run_file_url(run_name, name_substr, suffix=".table.json"):
    q = """query($e:String!,$p:String!,$r:String!){ project(name:$p,entityName:$e){
      run(name:$r){ files(first:200){ edges{ node{ name directUrl } } } } } }"""
    d = _gql(q, {"e": ENT, "p": PROJ, "r": run_name})
    for edge in d["project"]["run"]["files"]["edges"]:
        n = edge["node"]["name"]
        if name_substr in n and n.endswith(suffix):
            return edge["node"]["directUrl"]
    return None


def fetch_json(url):
    r = requests.get(url, timeout=120); r.raise_for_status()
    return r.json()


def fetch_csv(url):
    r = requests.get(url, timeout=120); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

# generating params, and the substrings that identify the matching FITTED columns
PARAMS = {
    "biasL": ["biasl", "bias_l", "bias"],
    "learn_rate": ["learn_rate", "learning_rate", "learn"],
    "softmax_temp": ["softmax_inverse_temperature", "softmax", "inverse_temp", "beta"],
}
GT_COL = {  # ground-truth column in groundtruth_params.csv
    "biasL": "param_biasL",
    "learn_rate": "param_learn_rate",
    "softmax_temp": "param_softmax_inverse_temperature",
}


def _find_col(df, needles, exclude=("true", "gt", "target", "r2", "error", "resid")):
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in exclude):
            continue
        if any(n in cl for n in needles):
            return c
    return None


def run_config_and_summary(run_name):
    q = """query($e:String!,$p:String!,$r:String!){ project(name:$p,entityName:$e){
      run(name:$r){ config summaryMetrics } } }"""
    d = _gql(q, {"e": ENT, "p": PROJ, "r": run_name})
    n = d["project"]["run"]
    return json.loads(n["config"]), json.loads(n["summaryMetrics"])


def cfg_get(cfg, key):
    v = cfg.get(key)
    return v.get("value") if isinstance(v, dict) else v


def load_table(run_name, name_substr="subject_fit_metrics"):
    """Fetch the subject_fit_metrics media-table JSON via signed URL (no SDK)."""
    url = run_file_url(run_name, name_substr, ".table.json")
    if url is None:
        raise FileNotFoundError(f"{name_substr} table not found for {run_name}")
    j = fetch_json(url)
    return pd.DataFrame(j["data"], columns=j["columns"])


def regen_gt(cfg):
    """Per-subject TRUE params = session-mean of the deterministically regenerated GT.
    Mirrors stage2_recovery.py: pure params-only regen from the run's logged data config.
    The full data config is logged as a NESTED dict under cfg['data'] (task/agent are
    themselves nested dicts the loader needs whole); the flat 'data.*' keys are absent.
    Requires the wrapper on sys.path (compute node)."""
    sys.path.insert(0, os.path.expanduser("~/code/aind-disrnn-wrapper/code"))
    sys.path.insert(0, os.path.join(os.getcwd(), "aind-disrnn-wrapper", "code"))
    os.environ.setdefault("DISRNN_GEN_WORKERS", "1")
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
    dc = cfg_get(cfg, "data")
    if not isinstance(dc, dict):
        raise RuntimeError("cfg['data'] is not a nested dict")
    ld = HierarchicalCognitiveAgents(
        task=dc["task"], agent=dc["agent"], num_trials=dc["num_trials"],
        num_subjects=dc["num_subjects"], num_sessions_per_subject=dc["num_sessions_per_subject"],
        eval_every_n=dc.get("eval_every_n", 2), batch_size=dc.get("batch_size"),
        batch_mode=dc.get("batch_mode", "random"),
        subject_seed_stride=dc.get("subject_seed_stride", 100000),
        generation_workers=1, seed=dc.get("seed", 42),
        heldout_session_mode=dc.get("heldout_session_mode", "interleaved"),
        heldout_frac=dc.get("heldout_frac", 0.2))
    gt = ld.groundtruth_table()
    return gt.groupby("subject_id")[list(GT_COL.values())].mean().reset_index()


if __name__ == "__main__":
    rows = []
    for N, rid in RUNS.items():
        cfg, summ = run_config_and_summary(rid)
        fit = load_table(rid)
        print(f"[{rid}] table cols ({fit.shape}): {list(fit.columns)}")
        gt = regen_gt(cfg)
        sid = _find_col(fit, ["subject_id", "subject_index", "subject"], exclude=())
        if fit[sid].dtype != object:  # index -> synth### to match gt subject_id
            fit = fit.copy(); fit[sid] = fit[sid].map(lambda i: f"synth{int(i):03d}")
        m = fit.merge(gt, left_on=sid, right_on="subject_id", how="inner")
        assert len(m) == N, f"merge got {len(m)} rows, expected {N}"
        row = {"num_subjects": N, "wandb_run_id": rid,
               "eval_likelihood": float(summ.get("eval_likelihood", np.nan))}
        r2s = []
        for p, needles in PARAMS.items():
            fc = _find_col(fit, needles)
            if fc is None:
                raise KeyError(f"no fitted column for {p} in {list(fit.columns)}")
            r2 = r2_score(m[GT_COL[p]].values, m[fc].values)
            row[f"r2_{p}"] = r2; r2s.append(r2)
        row["r2_mean"] = float(np.mean(r2s))
        rows.append(row)
        print(f"N={N} {rid}: biasL={row['r2_biasL']:.3f} learn={row['r2_learn_rate']:.3f} "
              f"softmax={row['r2_softmax_temp']:.3f} mean={row['r2_mean']:.3f}")
    out = pd.DataFrame(rows)[
        ["num_subjects", "r2_biasL", "r2_learn_rate", "r2_softmax_temp",
         "r2_mean", "eval_likelihood", "wandb_run_id"]]
    out.to_csv("stage2_baseline_recovery.csv", index=False)
    print("WROTE stage2_baseline_recovery.csv", out.shape)
