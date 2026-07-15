#!/usr/bin/env python
"""Compute stage-1 baseline_rl parameter-recovery R2 + Spearman -> stage1_baseline_recovery.csv.

No committed producer previously existed for this CSV (it predates the reproducibility push
that added compute_stage2_baseline_recovery.py for the stage-2 analog); this fills that gap
using the same cached fitted_params_per_subject JSON already used to freeze the CSV.

METHOD: baseline_rl fits an independent, correctly-specified ForagerQLearning per subject in
Stage 1 (static params across sessions), logging its fit as a `baseline_rl_output.json`
artifact with a `fitted_params_per_subject` dict keyed by subject_id. Recovery R2/Spearman
per param = r2_score / spearmanr(true_subject_static_value, fitted_value).

Canonical runs: N=50 f3ogvxjo, N=100 kbkxkyid, N=200 d2ljhfb8, N=300 jcf2s7ir.

Run anywhere with network access to the cached baseline_rl_output.json files (HPC scratch,
or re-fetched from the W&B run's logged artifact) and the master groundtruth_params.csv.

Output columns (frozen, keyed by wandb_run_id per the 'freeze the numbers' rule):
  num_subjects, r2_learn_rate, spearman_learn_rate, r2_biasL, spearman_biasL,
  r2_softmax_temp, spearman_softmax_temp, r2_mean, eval_likelihood, wandb_run_id
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

RUNS = {50: "f3ogvxjo", 100: "kbkxkyid", 200: "d2ljhfb8", 300: "jcf2s7ir"}
GT_COL = {"biasL": "param_biasL", "learn_rate": "param_learn_rate",
          "softmax_temp": "param_softmax_inverse_temperature"}
FIT_KEY = {"biasL": "biasL", "learn_rate": "learn_rate",
           "softmax_temp": "softmax_inverse_temperature"}


def load_fitted(output_json_path):
    d = json.load(open(output_json_path))
    fps = d["fitted_params_per_subject"]
    rows = []
    for sid, v in fps.items():
        row = {"subject_id": sid, "eval_likelihood": v.get("eval_likelihood")}
        row.update({f"fit_{k}": val for k, val in v["fitted_params"].items()})
        rows.append(row)
    return pd.DataFrame(rows), d.get("eval_likelihood", np.nan)


def load_true_static(gt_master_csv):
    """One static row per subject (Stage-1 params constant across sessions)."""
    gt = pd.read_csv(gt_master_csv)
    per_subj = (gt.groupby("subject_id")[list(GT_COL.values())]
                  .agg(lambda s: s.iloc[0]).reset_index())
    return per_subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-outputs-root", required=True,
                     help="Dir holding baseline_<run_id>/baseline_rl_output.json.")
    ap.add_argument("--gt-master", required=True)
    ap.add_argument("--out-dir", default="./analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    true_df = load_true_static(args.gt_master)
    rows = []
    for N, rid in RUNS.items():
        json_path = os.path.join(args.baseline_outputs_root, f"baseline_{rid}", "baseline_rl_output.json")
        fit, eval_lik = load_fitted(json_path)
        m = fit.merge(true_df, on="subject_id", how="inner")
        assert len(m) == N, f"merge got {len(m)} rows, expected {N}"
        row = {"num_subjects": N, "wandb_run_id": rid, "eval_likelihood": eval_lik}
        r2s = []
        for p in GT_COL:
            t = m[GT_COL[p]].values
            f = m[f"fit_{FIT_KEY[p]}"].values
            row[f"r2_{p}"] = r2_score(t, f)
            rho, _pval = spearmanr(t, f)
            row[f"spearman_{p}"] = float(rho)
            r2s.append(row[f"r2_{p}"])
        row["r2_mean"] = float(np.mean(r2s))
        rows.append(row)
        print(f"N={N} {rid}: biasL={row['r2_biasL']:.3f} learn={row['r2_learn_rate']:.3f} "
              f"softmax={row['r2_softmax_temp']:.3f} mean={row['r2_mean']:.3f} "
              f"spearman_biasL={row['spearman_biasL']:.3f} spearman_learn_rate={row['spearman_learn_rate']:.3f}")

    out = pd.DataFrame(rows)[
        ["num_subjects", "r2_learn_rate", "spearman_learn_rate", "r2_biasL", "spearman_biasL",
         "r2_softmax_temp", "spearman_softmax_temp", "r2_mean", "eval_likelihood", "wandb_run_id"]]
    out_csv = os.path.join(args.out_dir, "stage1_baseline_recovery.csv")
    out.to_csv(out_csv, index=False)
    print(f"WROTE {out_csv}", out.shape)


if __name__ == "__main__":
    main()
