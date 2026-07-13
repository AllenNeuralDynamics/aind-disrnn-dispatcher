#!/usr/bin/env python
"""Can the GRU's learned subject embedding predict a mouse's classical-RL parameters?

Issue #27. The wrapper's `run_analysis embedding-params` *visualizes* the subject
embedding coloured by each fitted baseline-RL parameter, but it does not quantify
anything. This script does the decoding: for each parameter, fit
`embedding (4-D) -> parameter` and report cross-validated R^2.

Two baselines are decoded because they carry DIFFERENT parameters, and the contrast
is the point:

  * CTT   (ForagerCompareThreshold): learn_rate, threshold, softmax_inverse_temperature, biasL
  * Bari  (ForagerQLearning L1F1_CK1): learn_rate, forget_rate_unchosen,
          choice_kernel_relative_weight, choice_kernel_step_size, biasL,
          softmax_inverse_temperature

If the embedding is a real cognitive coordinate rather than an arbitrary code, the
parameters both models share (learn_rate, biasL, inverse temperature) should decode
consistently from BOTH, and the model-specific ones (CTT's `threshold`, Bari's choice
kernel) tell us what else the embedding carries.

Controls (a raw R^2 here is easy to fool):
  * K-fold CV, so R^2 is out-of-sample.
  * A LABEL-SHUFFLE null: permute subjects, refit, repeat. The null is NOT centred on
    zero for small n / flexible models, so we report the shuffled distribution rather
    than assuming R^2 > 0 means anything.
  * A `curriculum_name` control: curriculum is a strong nuisance variable (it drives
    both the embedding and the RL fit). We report decoding from curriculum ALONE, so a
    parameter the embedding "predicts" only via curriculum is visible as such.

Inputs: the `subject_embedding_baseline_parameters.csv` written by
`run_analysis embedding-params` for each baseline.

Usage:
    python embedding_param_decode.py --ctt <dir> --bari <dir> [--out analysis/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
N_SPLITS = 5
N_SHUFFLE = 200
SEED = 0

EMBED_COLS = ["embedding_1", "embedding_2", "embedding_3", "embedding_4"]


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))


def _cv_r2(X, y, seed=SEED):
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(_ridge(), X, y, cv=cv, scoring="r2")))


def decode_one(df: pd.DataFrame, param: str, rng: np.random.Generator) -> dict:
    sub = df.dropna(subset=EMBED_COLS + [param, "curriculum_name"])
    X = sub[EMBED_COLS].to_numpy(float)
    y = sub[param].to_numpy(float)

    # --- Guard 1: a CONSTANT parameter is not decodable, it is fixed by the model.
    # Bari's `choice_kernel_step_size` is 1.0 for all 611 mice because
    # `choice_kernel="one_step"` fixes it by definition. Regressing on it yields a
    # degenerate R^2 of 1.0 (perfectly "predicting" a constant), which is meaningless.
    if np.nanstd(y) == 0 or len(np.unique(y)) <= 1:
        return dict(param=param, n=int(len(sub)), skipped="constant (fixed by the model, not fitted)",
                    constant_value=float(y[0]))

    # --- Guard 2: parameters that run to their fit bound produce runaway outliers, and
    # R^2 is variance-based so a handful of them dominate it. Bari's inverse temperature
    # hits its upper bound (100) for a few mice: median 4.25, max 100, skew 8.3. We
    # therefore report a RANK-based decoding (Spearman-style: R^2 on rank-transformed y)
    # alongside the raw one, and count the bound-pinned mice.
    at_bound = int((y >= 0.99 * y.max()).sum())
    skew = float(pd.Series(y).skew())
    y_rank = pd.Series(y).rank().to_numpy(float)

    r2 = _cv_r2(X, y)
    r2_rank = _cv_r2(X, y_rank)

    # curriculum-only control: how much is explained by the nuisance variable alone?
    C = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(
        sub[["curriculum_name"]])
    r2_curriculum = _cv_r2(C, y)

    # embedding + curriculum: does the embedding add anything ON TOP of curriculum?
    r2_both = _cv_r2(np.hstack([X, C]), y)

    # label-shuffle null, computed on the RANK metric (the robust one we report)
    null = []
    for _ in range(N_SHUFFLE):
        null.append(_cv_r2(X, rng.permutation(y_rank), seed=SEED))
    null = np.array(null)
    p = float((null >= r2_rank).mean())

    return dict(
        param=param, n=int(len(sub)),
        r2_embedding=r2,
        r2_embedding_rank=r2_rank,          # robust to bound-pinned outliers
        r2_curriculum_only=r2_curriculum,
        r2_embedding_plus_curriculum=r2_both,
        r2_added_over_curriculum=r2_both - r2_curriculum,
        n_at_fit_bound=at_bound, skew=skew,
        null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
        p_vs_shuffle=p,
    )


def run(csv: Path, label: str) -> list[dict]:
    df = pd.read_csv(csv)
    params = [c for c in df.columns
              if c not in EMBED_COLS + ["subject_index", "subject_id", "curriculum_name",
                                        "_subject_key", "_dot_id"]]
    rng = np.random.default_rng(SEED)
    print(f"\n=== {label}  (n={len(df)} subjects, {len(params)} parameters)")
    rows = []
    for p in params:
        r = decode_one(df, p, rng)
        r["model"] = label
        rows.append(r)
        if "skipped" in r:
            print(f"  {p:32} SKIPPED — {r['skipped']} (= {r['constant_value']})")
            continue
        sig = "***" if r["p_vs_shuffle"] < 0.001 else ("*" if r["p_vs_shuffle"] < 0.05 else "ns")
        bound = f"  [{r['n_at_fit_bound']} at fit bound, skew {r['skew']:.1f}]" \
            if r["n_at_fit_bound"] > 1 or abs(r["skew"]) > 2 else ""
        print(f"  {p:32} rank R2={r['r2_embedding_rank']:+.3f} "
              f"(raw {r['r2_embedding']:+.3f})  "
              f"curriculum-only {r['r2_curriculum_only']:+.3f}  "
              f"p={r['p_vs_shuffle']:.3f} {sig}{bound}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctt", required=True, type=Path)
    ap.add_argument("--bari", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    rows = []
    rows += run(args.ctt / "subject_embedding_baseline_parameters.csv", "ctt")
    rows += run(args.bari / "subject_embedding_baseline_parameters.csv", "bari")

    out = args.out / "embedding_param_decode.json"
    json.dump({"n_splits": N_SPLITS, "n_shuffle": N_SHUFFLE, "seed": SEED,
               "embedding_dims": len(EMBED_COLS), "results": rows},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
