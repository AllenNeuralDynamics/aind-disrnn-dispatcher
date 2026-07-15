#!/usr/bin/env python
"""Stage-4a PER-SESSION parameter recovery: baseline_rl (fixed fit broadcast to every
session) vs GRU session-blind vs GRU session-conditioned, within each true model FAMILY
(QLearning / CompareToThreshold / LossCounting). Mirrors
compute_stage3_persession_recovery.py -- same methods, extended to Stage 4a's per-family
parameter set. Unlike Stage 3, all three families here HAVE a correctly-specified fixed
baseline (that's the point of Stage 4a vs Stage 3's missing RW fitter).

Run on hpc-code (disrnn-cpu env, wrapper on sys.path, WANDB_API_KEY exported).

Per-family DRIFTING params (must match groundtruth_table param_* columns; only params that
actually carry a `drift` block in hierarchical_rl_stage4a.yaml are scored -- non-drifting
per-subject params like forget_rate_unchosen/choice_kernel_relative_weight/threshold have no
per-session ground-truth variation to recover and are intentionally excluded, same convention
as the static-param columns already reported in stage4a_recovery_combined.png panel b):
  QLearning:           biasL, learn_rate, softmax_inverse_temperature
  CompareToThreshold:  biasL, learn_rate, softmax_inverse_temperature
  LossCounting:        biasL, loss_count_threshold_mean

Output: stage4a_persession_recovery.csv (long format):
  true_family, cond [baseline_rl|gru_session_blind|gru_session_conditioned],
  param, r2, r2_raw, spearman, n_winsorized, n_session_rows, wandb_run_id
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
GRU_BLIND_RUN = "1xeoeclu"     # none, D4
GRU_COND_RUN = "mw1fhzt9"      # scalar, D4 -- same embed size as blind, for a fair pair
BASELINE_RUNIDS = {"QLearning": "2v4js1ld", "CompareToThreshold": "c03mqwwl",
                    "LossCounting": "qdgoftly"}
OUTBASE = "/home/han.hou/outputs/disrnn/wandb"

# Only params with an actual `drift` block in hierarchical_rl_stage4a.yaml -- these are the
# only ones with per-session ground-truth variation to recover.
FAMILY_PARAMS = {
    "QLearning": ["biasL", "learn_rate", "softmax_inverse_temperature"],
    # threshold has NO drift block (static per-subject, U[0.2,0.6]) -- included anyway
    # per user request, to see whether the STATIC param is even recoverable given the
    # broadcast; per-session "ground truth" is just the constant subject-level value
    # repeated across sessions, so this is really a subject-level recovery check dressed
    # in the per-session scoring machinery, not a drift-recovery result.
    "CompareToThreshold": ["biasL", "learn_rate", "softmax_inverse_temperature", "threshold"],
    "LossCounting": ["biasL", "loss_count_threshold_mean"],
}

# True generative parameter ranges (subject_param_dist in hierarchical_rl_stage4a.yaml),
# used to winsorize baseline_rl's occasional near-degenerate MLE fits -- same convention as
# stage2/stage2b/stage3.
WINSOR = {
    "biasL": 1.5, "learn_rate": 0.9, "softmax_inverse_temperature": 15.0,
    "loss_count_threshold_mean": 6.0, "threshold": 0.6,
}

DATA_CFG = {
    "task": {"type": "random_walk", "mean": 0, "seed": 42, "p_max": 1, "p_min": 0,
              "sigma": 0.15, "num_trials": 650, "reward_baiting": False},
    "agent": {
        "type": "mixture_family", "seed": 42,
        "loader_target": "data_loaders.hierarchical_synthetic.HierarchicalCognitiveAgents",
        "session_noise": {"biasL": 0.15},
        "subject_presets": {
            "assignment": "balanced",
            "presets": [
                {"name": "QLearning", "agent_class": "ForagerQLearning",
                 "agent_kwargs": {"number_of_learning_rate": 1, "number_of_forget_rate": 1,
                                   "choice_kernel": "one_step", "action_selection": "softmax"},
                 "agent_params": {},
                 "subject_param_dist": {
                     "learn_rate": {"type": "uniform", "min": 0.1, "max": 0.9},
                     "forget_rate_unchosen": {"type": "uniform", "min": 0.0, "max": 0.4},
                     "choice_kernel_relative_weight": {"type": "uniform", "min": 0.0, "max": 0.5},
                     "biasL": {"type": "uniform", "min": -1.5, "max": 1.5},
                     "softmax_inverse_temperature": {"type": "uniform", "min": 2.0, "max": 15.0}},
                 "drift": {"learn_rate": {"mode": "linear", "delta": 0.6},
                           "biasL": {"mode": "toward_zero", "frac": 0.8},
                           "softmax_inverse_temperature": {"mode": "sinusoidal", "amp": 4.0, "cycles": 1.0}},
                 "session_noise": {"learn_rate": 0.05, "biasL": 0.15, "softmax_inverse_temperature": 1.5}},
                {"name": "CompareToThreshold", "agent_class": "ForagerCompareThreshold",
                 "agent_kwargs": {"choice_kernel": "none"}, "agent_params": {},
                 "subject_param_dist": {
                     "learn_rate": {"type": "uniform", "min": 0.1, "max": 0.9},
                     "threshold": {"type": "uniform", "min": 0.2, "max": 0.6},
                     "softmax_inverse_temperature": {"type": "uniform", "min": 2.0, "max": 15.0},
                     "biasL": {"type": "uniform", "min": -1.5, "max": 1.5}},
                 "drift": {"learn_rate": {"mode": "linear", "delta": 0.6},
                           "biasL": {"mode": "toward_zero", "frac": 0.8},
                           "softmax_inverse_temperature": {"mode": "sinusoidal", "amp": 4.0, "cycles": 1.0}},
                 "session_noise": {"learn_rate": 0.05, "threshold": 0.03, "biasL": 0.15,
                                    "softmax_inverse_temperature": 1.5}},
                {"name": "LossCounting", "agent_class": "ForagerLossCounting",
                 "agent_kwargs": {}, "agent_params": {},
                 "subject_param_dist": {
                     "loss_count_threshold_mean": {"type": "uniform", "min": 1.0, "max": 6.0},
                     "loss_count_threshold_std": {"type": "uniform", "min": 0.2, "max": 2.0},
                     "biasL": {"type": "uniform", "min": -0.6, "max": 0.6}},
                 "drift": {"loss_count_threshold_mean": {"mode": "linear", "delta": 1.5},
                           "biasL": {"mode": "toward_zero", "frac": 0.8}},
                 "session_noise": {"loss_count_threshold_mean": 0.3, "biasL": 0.08}},
            ],
        },
    },
    "type": "synthetic_hierarchical", "batch_mode": "random", "batch_size": 512,
    "num_trials": 650, "eval_every_n": 2, "heldout_frac": 0.2, "num_subjects": 200,
    "subject_seed_stride": 100000, "heldout_session_mode": "tail",
    "num_sessions_per_subject": 40, "seed": 42,
}


def robust_r2(y_true, y_pred, param):
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
    dc = DATA_CFG
    ld = HierarchicalCognitiveAgents(
        task=dc["task"], agent=dc["agent"], num_trials=dc["num_trials"],
        num_subjects=dc["num_subjects"], num_sessions_per_subject=dc["num_sessions_per_subject"],
        eval_every_n=dc.get("eval_every_n", 2), batch_size=dc.get("batch_size"),
        subject_seed_stride=dc.get("subject_seed_stride", 100000), generation_workers=1,
        seed=dc.get("seed", 42), heldout_session_mode=dc.get("heldout_session_mode", "tail"),
        heldout_frac=dc.get("heldout_frac", 0.2))
    return ld.groundtruth_table(), dc


def load_baseline_fit(family, rid):
    g = glob.glob(f"{OUTBASE}/run-*-{rid}/files/outputs/baseline_rl_output.json")
    d = json.load(open(g[0]))
    return d["fitted_params_per_subject"]


def baseline_persession(gt, family):
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
        out.append({"true_family": family, "cond": "baseline_rl", "param": p, "r2": robust,
                     "r2_raw": raw, "spearman": rho, "n_winsorized": n_clip,
                     "n_session_rows": len(d["y_true"]), "wandb_run_id": BASELINE_RUNIDS[family]})
    return out


def dl(run_name, fname, dest):
    art = api.artifact(f"{ENT}/{PROJ}/gru-output-{run_name}:latest", type="training-output")
    return art.get_entry(fname).download(root=dest)


def gru_session_blind_persession(gt, run_name):
    rd = f"dl_{run_name}"
    os.makedirs(rd, exist_ok=True)
    dl(run_name, "subject_embeddings.pkl", rd)
    import pickle
    emb = pickle.load(open(os.path.join(rd, "subject_embeddings.pkl"), "rb"))
    emb_df = emb if isinstance(emb, pd.DataFrame) else pd.DataFrame(emb)
    ecols = [c for c in emb_df.columns if c.startswith("embedding_")]
    subj_fam = gt.groupby("subject_id").agg(fam=("preset_name", "first")).reset_index()
    out = []
    for family in FAMILY_PARAMS:
        fam_subj = subj_fam[subj_fam.fam == family].subject_id.values
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
            out.append({"true_family": family, "cond": "gru_session_blind", "param": p, "r2": r2,
                         "r2_raw": r2, "spearman": np.nan, "n_winsorized": 0,
                         "n_session_rows": len(y_true), "wandb_run_id": run_name})
    return out


def gru_session_conditioned_persession(gt, run_name):
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
    subj_fam = gt.groupby("subject_id").agg(fam=("preset_name", "first")).reset_index()
    out = []
    for family in FAMILY_PARAMS:
        fam_subj = subj_fam[subj_fam.fam == family].subject_id.values
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
            out.append({"true_family": family, "cond": "gru_session_conditioned", "param": p, "r2": r2,
                         "r2_raw": r2, "spearman": np.nan, "n_winsorized": 0,
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

    rows += gru_session_conditioned_persession(gt, GRU_COND_RUN)
    print("GRU session-conditioned done")

    out = pd.DataFrame(rows)
    out.to_csv("stage4a_persession_recovery.csv", index=False)
    print("WROTE stage4a_persession_recovery.csv", out.shape)
    print(out.to_string())
