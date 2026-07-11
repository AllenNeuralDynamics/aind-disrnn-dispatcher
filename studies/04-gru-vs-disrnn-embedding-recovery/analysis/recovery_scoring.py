"""Embedding-recovery scoring for the disRNN/GRU subject-embedding study.

Two-part recovery score (model-agnostic; works for GRU, disRNN, and — via the
absolute-likelihood key — baseline_rl):

  (1) FIT: likelihood_relative_to_groundtruth  (model NL / generating-policy NL,
      ceiling 1.0).  Pulled from W&B; baseline_rl's is computed as
      eval_likelihood / groundtruth_likelihood.

  (2) RECOVERY: how well the learned subject embedding encodes the TRUE
      generating parameters (biasL, learn_rate, softmax_inverse_temperature).
      - CCA: canonical correlations between embedding (N x d_emb) and true
        params (N x 3).  Reports mean and per-component canonical r.
      - Ridge R^2: cross-validated R^2 of predicting each true param from the
        embedding (embedding -> param).  This is the interpretable "can I read
        parameter X off the embedding?" score, robust to d_emb != 3.

The true per-subject params are STATIC in Stage 1 (one value per subject across
sessions) and deterministic in subject_idx alone, so a single master
groundtruth table serves every run.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score

PARAM_COLS = ["param_biasL", "param_learn_rate", "param_softmax_inverse_temperature"]
PARAM_SHORT = {"param_biasL": "biasL",
               "param_learn_rate": "learn_rate",
               "param_softmax_inverse_temperature": "softmax_temp"}


def load_true_params(gt_master_csv: str) -> pd.DataFrame:
    """One static row per subject (params constant across sessions in Stage 1)."""
    gt = pd.read_csv(gt_master_csv)
    per_subj = (gt.groupby("subject_id")[PARAM_COLS]
                  .agg(lambda s: s.iloc[0])
                  .reset_index())
    # sanity: params really are static across sessions
    spread = gt.groupby("subject_id")[PARAM_COLS].nunique()
    assert (spread == 1).all().all(), "ground-truth params vary within subject (not Stage-1 static)"
    return per_subj


def load_embedding(emb_pickle: str) -> pd.DataFrame:
    df = pd.read_pickle(emb_pickle)
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    return df[["subject_id"] + emb_cols].copy(), emb_cols


def align(emb_df: pd.DataFrame, true_df: pd.DataFrame, emb_cols):
    m = emb_df.merge(true_df, on="subject_id", how="inner")
    assert len(m) == len(emb_df), f"alignment lost rows: {len(m)} vs {len(emb_df)}"
    E = m[emb_cols].to_numpy(float)
    P = m[PARAM_COLS].to_numpy(float)
    return E, P, m


def cca_scores(E, P):
    """Canonical correlations between embedding and true params."""
    k = min(E.shape[1], P.shape[1])
    Es = StandardScaler().fit_transform(E)
    Ps = StandardScaler().fit_transform(P)
    cca = CCA(n_components=k, max_iter=1000)
    U, V = cca.fit_transform(Es, Ps)
    rs = [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)]
    rs = [abs(r) for r in rs]
    return {"cca_r_mean": float(np.mean(rs)),
            "cca_r_top": float(rs[0]),
            "cca_r_all": [float(r) for r in rs]}


def ridge_r2(E, P, n_splits=5, alpha=1.0, seed=0):
    """Cross-validated R^2 predicting each true param from the embedding."""
    n = E.shape[0]
    n_splits = min(n_splits, n)
    Es = StandardScaler().fit_transform(E)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for j, col in enumerate(PARAM_COLS):
        y = P[:, j]
        pred = cross_val_predict(Ridge(alpha=alpha), Es, y, cv=kf)
        out[PARAM_SHORT[col]] = float(r2_score(y, pred))
    out["r2_mean"] = float(np.mean(list(out.values())))
    return out


def score_run(emb_pickle: str, true_df: pd.DataFrame) -> dict:
    emb_df, emb_cols = load_embedding(emb_pickle)
    E, P, _ = align(emb_df, true_df, emb_cols)
    res = {"n_subjects": E.shape[0], "emb_dim": E.shape[1]}
    res.update(cca_scores(E, P))
    r2 = ridge_r2(E, P)
    res.update({f"r2_{k}": v for k, v in r2.items()})
    return res
