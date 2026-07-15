#!/usr/bin/env python
"""Stage-3 PER-SESSION parameter recovery: baseline_rl (fixed fit broadcast to every
session) vs GRU session-blind vs GRU session-conditioned, within each true model family
(Bari2019 / Hattori2019 / RescorlaWagner). Mirrors compute_stage2_baseline_persession.py /
compute_stage2b_baseline_persession.py + the GRU session-conditioned reconstruction already
used for stage2/stage2b trajectory figures -- same methods, extended to Stage 3's per-family
parameter set (each preset has its own param subset; RW has no fixed baseline available).

Run on hpc-code (disrnn-cpu env, wrapper on sys.path, WANDB_API_KEY exported).

Per-family PARAMS (must match groundtruth_table param_* columns):
  Bari2019:    biasL, choice_kernel_relative_weight, forget_rate_unchosen, learn_rate,
               softmax_inverse_temperature
  Hattori2019: biasL, learn_rate_rew, learn_rate_unrew, softmax_inverse_temperature
  RescorlaWagner: biasL, learn_rate, epsilon

Output: stage3_persession_recovery.csv (long format):
  true_preset, cond [baseline_rl|gru_session_blind|gru_session_conditioned],
  param, r2, n_session_rows, wandb_run_id
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from scipy.stats import spearmanr

sys.path.insert(0, os.path.expanduser("~/scratch/recovery-smoke/aind-disrnn-wrapper/code"))
from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
from utils.multisubject import compute_session_conditioned_context_dataframe
import wandb

ENT, PROJ = "AIND-disRNN", "embedding_recovery"
GRU_BLIND_RUN = "1y7vz70o"     # none, D4
GRU_COND_RUN = "ok01hebs"      # scalar, D4 -- same embed size as blind, for a fair pair
BASELINE_RUNIDS = {"Bari2019": "x548cbk7", "Hattori2019": "cthtvmln"}  # no RW baseline exists
OUTBASE = "/home/han.hou/outputs/disrnn/wandb"

FAMILY_PARAMS = {
    "Bari2019": ["biasL", "choice_kernel_relative_weight", "forget_rate_unchosen",
                 "learn_rate", "softmax_inverse_temperature"],
    "Hattori2019": ["biasL", "learn_rate_rew", "learn_rate_unrew", "softmax_inverse_temperature"],
    "RescorlaWagner": ["biasL", "learn_rate", "epsilon"],
}

# True generative parameter ranges (from hierarchical_rl_stage3.yaml subject_param_dist),
# used to winsorize baseline_rl's occasional near-degenerate MLE fits (same identifiability
# issue as stage2/stage2b's softmax_inverse_temperature, here affecting several params).
# Winsor threshold = true max (fits cannot know the true range, but overshoots this far
# beyond it are fit failures, not real signal).
WINSOR = {
    "biasL": 1.5, "learn_rate": 0.9, "forget_rate_unchosen": 0.4,
    "softmax_inverse_temperature": 15.0, "choice_kernel_relative_weight": 0.5,
    "learn_rate_rew": 0.9, "learn_rate_unrew": 0.6, "epsilon": 0.35,
}


def robust_r2(y_true, y_pred, param):
    """Winsorize y_pred at the true ceiling for this param (if one is known), report
    raw + robust R2 + Spearman rank correlation -- same convention as stage2/stage2b's
    baseline softmax_temp handling."""
    from scipy.stats import spearmanr
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    raw = r2_score(y_true, y_pred)
    thr = WINSOR.get(param)
    if thr is None:
        return raw, raw, np.nan, 0
    y_clip = np.clip(y_pred, None, thr)
    n_clipped = int((y_pred > thr).sum())
    robust = r2_score(y_true, y_clip)
    rho = spearmanr(y_true, y_pred).correlation
    return raw, robust, rho, n_clipped

api = wandb.Api()


def regen_gt():
    inv = json.load(open(os.path.expanduser("~/scratch/recovery-smoke/stage3_inventory.json")))
    dc = next(m["data_cfg"] for m in inv.values() if m.get("state") == "finished")
    ld = HierarchicalCognitiveAgents(
        task=dc["task"], agent=dc["agent"], num_trials=dc["num_trials"],
        num_subjects=dc["num_subjects"], num_sessions_per_subject=dc["num_sessions_per_subject"],
        eval_every_n=dc.get("eval_every_n", 2), batch_size=dc.get("batch_size"),
        subject_seed_stride=dc.get("subject_seed_stride", 100000), generation_workers=1,
        seed=dc.get("seed", 42), heldout_session_mode=dc.get("heldout_session_mode", "tail"),
        heldout_frac=dc.get("heldout_frac", 0.2))
    return ld.groundtruth_table(), dc


def load_baseline_fit(fam, rid):
    g = glob.glob(f"{OUTBASE}/run-*-{rid}/files/outputs/baseline_rl_output.json")
    d = json.load(open(g[0]))
    return d["fitted_params_per_subject"]


def baseline_persession(gt, family):
    """Broadcast the fixed per-subject fit to every session of that subject; score
    against per-session ground truth (only meaningful within the matched family)."""
    fit = load_baseline_fit(family, BASELINE_RUNIDS[family])
    sub = gt[gt.preset_name == family].copy()
    rows_by_param = {p: {"y_true": [], "y_pred": []} for p in FAMILY_PARAMS[family]}
    for sid, rec in fit.items():
        if sid not in sub.subject_id.values:
            continue
        fp = rec["fitted_params"]
        srows = sub[sub.subject_id == sid]
        for p in FAMILY_PARAMS[family]:
            col = f"param_{p}"
            if col not in srows.columns or p not in fp:
                continue
            rows_by_param[p]["y_true"].extend(srows[col].values)
            rows_by_param[p]["y_pred"].extend([fp[p]] * len(srows))
    out = []
    for p, d in rows_by_param.items():
        if len(d["y_true"]) == 0:
            continue
        raw, robust, rho, n_clip = robust_r2(d["y_true"], d["y_pred"], p)
        out.append({"true_preset": family, "cond": "baseline_rl", "param": p, "r2": robust,
                     "r2_raw": raw, "spearman": rho, "n_winsorized": n_clip,
                     "n_session_rows": len(d["y_true"]), "wandb_run_id": BASELINE_RUNIDS[family]})
    return out


def dl(run_name, fname, dest):
    art = api.artifact(f"{ENT}/{PROJ}/gru-output-{run_name}:latest", type="training-output")
    return art.get_entry(fname).download(root=dest)


def gru_session_blind_persession(gt, run_name):
    """Session-blind: ONE embedding per subject (no session conditioning) -> broadcast
    its within-family-regressed prediction to every session. Uses the same
    cross-validated linear-regression method as stage3_recovery.py's within-family R2,
    but scored against PER-SESSION ground truth instead of the per-subject mean."""
    rd = f"dl_{run_name}"
    os.makedirs(rd, exist_ok=True)
    dl(run_name, "subject_embeddings.pkl", rd)
    import pickle
    emb = pickle.load(open(os.path.join(rd, "subject_embeddings.pkl"), "rb"))
    emb_df = emb if isinstance(emb, pd.DataFrame) else pd.DataFrame(emb)
    ecols = [c for c in emb_df.columns if c.startswith("embedding_")]
    subj_preset = gt.groupby("subject_id").agg(preset_name=("preset_name", "first")).reset_index()
    out = []
    for family in FAMILY_PARAMS:
        fam_subj = subj_preset[subj_preset.preset_name == family].subject_id.values
        fam_emb = emb_df[emb_df.subject_id.isin(fam_subj)].copy()
        subj_mean = gt[gt.subject_id.isin(fam_subj)].groupby("subject_id")[
            [f"param_{p}" for p in FAMILY_PARAMS[family]]].mean().reset_index()
        m = fam_emb.merge(subj_mean, on="subject_id")
        X = m[ecols].values
        groups = m["subject_id"].values
        for p in FAMILY_PARAMS[family]:
            y = m[f"param_{p}"].values
            gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            yhat_subj = cross_val_predict(LinearRegression(), X, y, groups=groups, cv=gkf)
            pred_by_subj = dict(zip(m.subject_id.values, yhat_subj))
            sess = gt[gt.subject_id.isin(fam_subj)][["subject_id", f"param_{p}"]].dropna()
            y_true = sess[f"param_{p}"].values
            y_pred = sess.subject_id.map(pred_by_subj).values
            r2 = r2_score(y_true, y_pred)
            out.append({"true_preset": family, "cond": "gru_session_blind", "param": p, "r2": r2,
                         "r2_raw": r2, "spearman": spearmanr(y_true, y_pred).correlation, "n_winsorized": 0,
                         "n_session_rows": len(y_true), "wandb_run_id": run_name})
    return out


def gru_session_conditioned_persession(gt, run_name, meta):
    """Session-conditioned: use compute_session_conditioned_context_dataframe to get a
    per-(subject,session) embedding, then within-family cross-validated regression scored
    per session -- the actual per-session prediction, not a broadcast subject value."""
    rd = f"dl_{run_name}"
    os.makedirs(rd, exist_ok=True)
    P = json.load(open(dl(run_name, "params.json", rd)))
    md = json.load(open(dl(run_name, "multisubject_metadata.json", rd)))
    arch = json.load(open(dl(run_name, "gru_config.json", rd)))["architecture"]
    ctx = compute_session_conditioned_context_dataframe(
        P, session_context=md["session_context"],
        session_encoding_type=arch["session_encoding_type"],
        session_integration_type=arch.get("session_integration_type", "direct"),
        session_fourier_k=int(arch.get("session_fourier_k", 4)),
        session_delta_n_layers=int(arch["session_delta_n_layers"]),
        session_delta_hidden_size=int(arch["session_delta_hidden_size"]),
        session_curriculum_lambda=1.0,
        session_max_index_by_subject_index=md["session_max_index_by_subject_index"],
        train_session_ids=md.get("train_session_ids"), eval_session_ids=md.get("eval_session_ids"),
        selected_subject_indices=None)
    ecols = [c for c in ctx.columns if c.startswith("embedding_")]
    subj_preset = gt.groupby("subject_id").agg(preset_name=("preset_name", "first")).reset_index()
    out = []
    for family in FAMILY_PARAMS:
        fam_subj = subj_preset[subj_preset.preset_name == family].subject_id.values
        fam_ctx = ctx[ctx.subject_id.isin(fam_subj)].copy()
        m = fam_ctx.merge(gt[["subject_id", "session_id"] + [f"param_{p}" for p in FAMILY_PARAMS[family]]],
                           on=["subject_id", "session_id"], how="inner")
        X = m[ecols].values
        groups = m["subject_id"].values
        for p in FAMILY_PARAMS[family]:
            y = m[f"param_{p}"].values
            valid = ~pd.isna(y)
            gkf = GroupKFold(n_splits=min(5, len(np.unique(groups[valid]))))
            yhat = cross_val_predict(LinearRegression(), X[valid], y[valid], groups=groups[valid], cv=gkf)
            r2 = r2_score(y[valid], yhat)
            out.append({"true_preset": family, "cond": "gru_session_conditioned", "param": p, "r2": r2,
                         "r2_raw": r2, "spearman": spearmanr(y[valid], yhat).correlation, "n_winsorized": 0,
                         "n_session_rows": int(valid.sum()), "wandb_run_id": run_name})
    return out


if __name__ == "__main__":
    gt, dc = regen_gt()
    print("groundtruth_table:", gt.shape)

    rows = []
    for fam in BASELINE_RUNIDS:
        rows += baseline_persession(gt, fam)
        print(f"baseline {fam} done")

    rows += gru_session_blind_persession(gt, GRU_BLIND_RUN)
    print("GRU session-blind done")

    rows += gru_session_conditioned_persession(gt, GRU_COND_RUN, dc)
    print("GRU session-conditioned done")

    out = pd.DataFrame(rows)
    out.to_csv("stage3_persession_recovery.csv", index=False)
    print("WROTE stage3_persession_recovery.csv", out.shape)
    print(out.to_string())
