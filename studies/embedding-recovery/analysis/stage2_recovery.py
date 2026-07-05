
"""Stage-2 SUBJECT-level recovery analysis (runs on HPC: reaches W&B GCS + regenerates GT).
For each GRU run: download subject_embeddings.pkl from its W&B training-output
artifact; regenerate ground truth byte-identically from its logged config; compute
per-subject param recovery (R2 + canonical-corr) from the SUBJECT embedding, using
each subject's session-MEAN parameters as the target.

Session-TRAJECTORY recovery (per-session embedding vs drifting per-session params,
using the learned session-delta MLP) is a SEPARATE script: stage2_session_traj.py.
"""
import os, sys, json, pickle, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/scratch/recovery-smoke/aind-disrnn-wrapper/code"))
os.environ.setdefault("DISRNN_GEN_WORKERS","8")

import wandb
api = wandb.Api()
ENT, PROJ = "AIND-disRNN", "embedding_recovery"
inv = json.load(open(sys.argv[1]))
outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)

from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

PARAMS = ["param_biasL","param_learn_rate","param_softmax_inverse_temperature"]

def regen_gt(data_cfg):
    # FAST params-only regen (no trial simulation, no multiprocessing).
    ld = HierarchicalCognitiveAgents(
        task=data_cfg["task"], agent=data_cfg["agent"],
        num_trials=data_cfg["num_trials"], num_subjects=data_cfg["num_subjects"],
        num_sessions_per_subject=data_cfg["num_sessions_per_subject"],
        eval_every_n=data_cfg.get("eval_every_n",2),
        batch_size=data_cfg.get("batch_size"), batch_mode=data_cfg.get("batch_mode","random"),
        subject_seed_stride=data_cfg.get("subject_seed_stride",100000),
        generation_workers=1, seed=data_cfg.get("seed",42),
        heldout_session_mode=data_cfg.get("heldout_session_mode","interleaved"),
        heldout_frac=data_cfg.get("heldout_frac",0.2),
    )
    return ld.groundtruth_table()

def dl_artifact_file(run_name, fname, dest):
    art = api.artifact(f"{ENT}/{PROJ}/gru-output-{run_name}:latest", type="training-output")
    p = art.get_entry(fname).download(root=dest)
    return p

def subj_recovery(emb_df, gt_df):
    # per-subject TRUE params = session-mean (subject centroid proxy under drift)
    truemean = gt_df.groupby("subject_id")[PARAMS].mean().reset_index()
    m = emb_df.merge(truemean, on="subject_id", how="inner")
    ecols = [c for c in emb_df.columns if c.startswith("embedding_")]
    X = m[ecols].values
    out = {}
    for p in PARAMS:
        y = m[p].values
        yhat = cross_val_predict(LinearRegression(), X, y, cv=5)
        out[p] = r2_score(y, yhat)
    # CCA top corr
    k = min(len(ecols), len(PARAMS))
    cca = CCA(n_components=k); 
    Xc,Yc = cca.fit_transform(X, m[PARAMS].values)
    ccorr = [float(np.corrcoef(Xc[:,i],Yc[:,i])[0,1]) for i in range(k)]
    out["cca_top"] = max(ccorr); out["n"] = len(m)
    return out

results = []
for name, meta in inv.items():
    if meta["stage"] != "stage2" or meta["state"] != "finished": continue
    rd = os.path.join(outdir, name); os.makedirs(rd, exist_ok=True)
    try:
        dl_artifact_file(name, "subject_embeddings.pkl", rd)
        with open(os.path.join(rd,"subject_embeddings.pkl"),"rb") as f:
            emb = pickle.load(f)
        emb_df = emb if isinstance(emb, pd.DataFrame) else pd.DataFrame(emb)
        gt = regen_gt(meta["data_cfg"])
        rec = subj_recovery(emb_df, gt)
        rec.update({"run":name,"enc":meta["enc"],"num_subjects":meta["num_subjects"],
                    "lik_rel":meta["lik_rel"],
                    "R2_mean": float(np.mean([rec[p] for p in PARAMS]))})
        results.append(rec)
        print(f"{name} enc={meta['enc']} N={meta['num_subjects']} R2mean={rec['R2_mean']:.3f} cca={rec['cca_top']:.3f}")
    except Exception as ex:
        print(f"{name} FAILED: {type(ex).__name__}: {ex}")
df = pd.DataFrame(results)
df.to_csv(os.path.join(outdir,"stage2_subject_recovery.csv"), index=False)
print("WROTE", os.path.join(outdir,"stage2_subject_recovery.csv"), df.shape)
