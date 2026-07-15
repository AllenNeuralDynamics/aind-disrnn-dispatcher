
"""Stage-2 SESSION-TRAJECTORY recovery using the TRAINING CODE's own reconstruction.
Calls utils.multisubject.compute_session_conditioned_context_dataframe (the same fn
that backs the checkpoint subject_session_context_state_space.png plots) to get the
per-(subject,session) session-conditioned embedding, then tests whether that trajectory
recovers the DRIFTING per-session parameters. No reimplementation.
"""
import os, sys, json, numpy as np, pandas as pd
STAGE=os.environ.get("STAGE","stage2")
sys.path.insert(0, os.path.expanduser("~/scratch/recovery-smoke/aind-disrnn-wrapper/code"))
import wandb
api = wandb.Api(); ENT,PROJ="AIND-disRNN","embedding_recovery"
inv=json.load(open(sys.argv[1])); outdir=sys.argv[2]; os.makedirs(outdir,exist_ok=True)
from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
from utils.multisubject import compute_session_conditioned_context_dataframe
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pickle
PARAMS=["param_biasL","param_learn_rate","param_softmax_inverse_temperature"]
# Spearman rho reported alongside R2 (same y_true/y_pred pair) for all three drifting
# params, mirroring the convention used for the baseline_rl per-session CSVs.
SPEARMAN_PARAMS=["param_biasL","param_learn_rate","param_softmax_inverse_temperature"]

def regen_gt(cfg):
    ld=HierarchicalCognitiveAgents(task=cfg["task"],agent=cfg["agent"],num_trials=cfg["num_trials"],
        num_subjects=cfg["num_subjects"],num_sessions_per_subject=cfg["num_sessions_per_subject"],
        eval_every_n=cfg.get("eval_every_n",2),batch_size=cfg.get("batch_size"),
        batch_mode=cfg.get("batch_mode","random"),subject_seed_stride=cfg.get("subject_seed_stride",100000),
        generation_workers=1,seed=cfg.get("seed",42),
        heldout_session_mode=cfg.get("heldout_session_mode","interleaved"),heldout_frac=cfg.get("heldout_frac",0.2))
    return ld.groundtruth_table()

def fetch(run,fn,dest):
    a=api.artifact(f"{ENT}/{PROJ}/gru-output-{run}:latest",type="training-output")
    return a.get_entry(fn).download(root=dest)

rows=[]
for name,meta in inv.items():
    if meta["stage"]!=STAGE or meta["state"]!="finished" or meta["enc"]!="scalar": continue
    rd=os.path.join(outdir,name); os.makedirs(rd,exist_ok=True)
    try:
        P=pickle.load(open(fetch(name,"params.pkl",rd),"rb")) if False else json.load(open(fetch(name,"params.json",rd)))
        md=json.load(open(fetch(name,"multisubject_metadata.json",rd)))
        arch=json.load(open(fetch(name,"gru_config.json",rd)))["architecture"]
        # canonical reconstruction: ALL subjects (selected_subject_indices=None)
        ctx=compute_session_conditioned_context_dataframe(
            P, session_context=md["session_context"],
            session_encoding_type=arch["session_encoding_type"],
            session_integration_type=arch.get("session_integration_type","direct"),
            session_fourier_k=int(arch.get("session_fourier_k",4)),
            session_delta_n_layers=int(arch["session_delta_n_layers"]),
            session_delta_hidden_size=int(arch["session_delta_hidden_size"]),
            session_curriculum_lambda=1.0,
            session_max_index_by_subject_index=md["session_max_index_by_subject_index"],
            train_session_ids=md.get("train_session_ids"), eval_session_ids=md.get("eval_session_ids"),
            selected_subject_indices=None)
        # ctx has subject_id, session_id + embedding_* per (subject,session)
        gt=regen_gt(meta["data_cfg"])
        # merge on (subject_id, session_id)
        gt_key=gt[["subject_id","session_id","session_frac"]+PARAMS].copy()
        m=ctx.merge(gt_key, on=["subject_id","session_id"], how="inner")
        ecols=[c for c in ctx.columns if c.startswith("embedding_")]
        # SESSION-CONDITIONED embedding = subject_emb + learned session delta
        X=m[ecols].values; groups=m["subject_id"].values
        # ABLATION: subject-only embedding (delta zeroed) = one point per subject,
        # broadcast to all its sessions. Isolates what the learned session delta adds.
        subj_emb=np.array(json.load(open(os.path.join(rd,"params.json")))["multisubject_gru"]["subject_embeddings"]) \
                 if os.path.exists(os.path.join(rd,"params.json")) else None
        s2i=md["subject_id_to_index"]
        Xsub=np.array([subj_emb[s2i[sid]] for sid in m["subject_id"]])
        gkf=GroupKFold(n_splits=5)
        out={"run":name,"N":meta["num_subjects"],"n_rows":len(m),"n_emb":len(ecols)}
        for p in PARAMS:
            yhat=cross_val_predict(LinearRegression(),X,m[p].values,cv=gkf,groups=groups)
            out[f"sess_R2_{p}"]=r2_score(m[p].values,yhat)
            yhat_s=cross_val_predict(LinearRegression(),Xsub,m[p].values,cv=gkf,groups=groups)
            out[f"subjonly_R2_{p}"]=r2_score(m[p].values,yhat_s)
            if p in SPEARMAN_PARAMS:
                out[f"sess_spearman_{p}"]=float(spearmanr(m[p].values,yhat)[0])
                out[f"subjonly_spearman_{p}"]=float(spearmanr(m[p].values,yhat_s)[0])
        yhat_f=cross_val_predict(LinearRegression(),X,m["session_frac"].values,cv=gkf,groups=groups)
        out["sessfrac_R2"]=r2_score(m["session_frac"].values,yhat_f)
        yhat_fs=cross_val_predict(LinearRegression(),Xsub,m["session_frac"].values,cv=gkf,groups=groups)
        out["sessfrac_R2_subjonly"]=r2_score(m["session_frac"].values,yhat_fs)
        out["sess_R2_mean"]=float(np.mean([out[f"sess_R2_{p}"] for p in PARAMS]))
        out["subjonly_R2_mean"]=float(np.mean([out[f"subjonly_R2_{p}"] for p in PARAMS]))
        rows.append(out); m.to_csv(os.path.join(rd,"ctx_merged.csv"),index=False)
        print(f"{name} N={meta['num_subjects']} rows={len(m)} sess_R2_mean={out['sess_R2_mean']:.3f} (subjonly {out['subjonly_R2_mean']:.3f}) sessfrac_R2={out['sessfrac_R2']:.3f} (subjonly {out['sessfrac_R2_subjonly']:.3f})")
    except Exception as ex:
        import traceback; print(f"{name} FAILED: {type(ex).__name__}: {ex}"); traceback.print_exc()
df=pd.DataFrame(rows); df.to_csv(os.path.join(outdir,f"{STAGE}_session_trajectory_recovery.csv"),index=False)
print("WROTE",df.shape)
