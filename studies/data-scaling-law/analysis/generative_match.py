#!/usr/bin/env python
"""Generative behavioral-match vs D (2nd-order validation).

Rolls-out GRU as an agent; compares the switch-triggered behavioral curve (post-switch
choice by reward x run-length) of model vs real mouse. Headline = subject-mean correlation
(the corr~0.96 result) + subject-balanced RMSE, on the COMBINED session partition.

Pulls W&B groups generative-v{1,2}@... (one run per source ratio x seed), dedups to the
newest run per (variant, ratio, seed), reads the logged combined-partition scalars, averages
over seeds, and plots match-vs-D for v1 (SC off) and v2 (SC active). Run with the wrapper venv.

NOTE: behavioral match is logged for the COMBINED partition (all sessions). The per-partition
train/eval breakdowns exist too but combined is the headline (largest n, most stable).
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
PREFIXES = ("generative-v1@", "generative-v2@")
DLAB = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}

# the run-length-resolved switch curve (the corr~0.96 headline stat)
STAT = "post_switch_by_reward_and_run_length"
K_CORR = f"combined/switch_triggered/quantitative_summary/subject_mean/{STAT}/correlation"
K_MSE = (f"combined/switch_triggered/delta_significance_summary/{STAT}/"
         "subject_balanced_error_summary/mean_squared_error")
K_CORR_OVERALL = "combined/switch_triggered/quantitative_summary/subject_mean/overall/correlation"


def _variant(meta):
    v = (meta or {}).get("variant", "")
    return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)


def collect():
    api = wandb.Api()
    runs = [r for r in api.runs(PROJECT)
            if any((r.group or "").startswith(p) for p in PREFIXES) and r.state == "finished"]
    by_cell = {}
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        var = _variant(meta)
        ratio = meta.get("source_subject_ratio"); seed = meta.get("source_seed")
        if var is None or ratio is None:
            continue
        key = (var, round(float(ratio), 3), seed)
        prev = by_cell.get(key)
        if prev is None or str(getattr(r, "created_at", "")) > str(getattr(prev, "created_at", "")):
            by_cell[key] = r
    print(f"  {len(runs)} finished generative runs -> {len(by_cell)} cells")
    rows = []
    for (var, ratio, seed), r in by_cell.items():
        s = r.summary
        corr = s.get(K_CORR); mse = s.get(K_MSE); corr_o = s.get(K_CORR_OVERALL)
        if corr is None or mse is None:
            print(f"    [skip {r.name[:40]}] missing combined scalars")
            continue
        rows.append(dict(variant=var, ratio=ratio, D=DLAB.get(ratio, ratio), seed=seed,
                         corr=float(corr), corr_overall=float(corr_o) if corr_o is not None else None,
                         rmse=math.sqrt(float(mse))))
    return rows


def main():
    rows = collect()
    print(f"collected {len(rows)} generative cells")
    # aggregate over seeds per (variant, D)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["variant"], r["D"])]["corr"].append(r["corr"])
        agg[(r["variant"], r["D"])]["rmse"].append(r["rmse"])
    out = {}
    for (var, D), m in agg.items():
        out[f"{var}_D{D}"] = dict(D=D, n_seeds=len(m["corr"]),
                                  corr_mean=float(np.mean(m["corr"])), corr_sd=float(np.std(m["corr"])),
                                  rmse_mean=float(np.mean(m["rmse"])), rmse_sd=float(np.std(m["rmse"])))
    json.dump(out, open(HERE / "generative_match.json", "w"), indent=2)

    Ds = sorted({v["D"] for v in out.values()})
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(11, 4.3))
    for var, mk in (("v1", "o-"), ("v2", "s-")):
        xs = [D for D in Ds if f"{var}_D{D}" in out]
        cm = [out[f"{var}_D{D}"]["corr_mean"] for D in xs]
        cs = [out[f"{var}_D{D}"]["corr_sd"] for D in xs]
        rm = [out[f"{var}_D{D}"]["rmse_mean"] for D in xs]
        rs = [out[f"{var}_D{D}"]["rmse_sd"] for D in xs]
        lab = "v1 (SC off)" if var == "v1" else "v2 (SC active)"
        axc.errorbar(xs, cm, yerr=cs, fmt=mk, capsize=3, label=lab)
        axr.errorbar(xs, rm, yerr=rs, fmt=mk, capsize=3, label=lab)
    for ax in (axc, axr):
        ax.set_xscale("log"); ax.set_xlabel("# training mice (D)"); ax.legend()
    axc.set_ylabel("subject-mean correlation (model vs animal)")
    axc.set_title("Generative switch-curve match vs D")
    axr.set_ylabel("subject-balanced RMSE")
    axr.set_title("Generative switch-curve error vs D")
    fig.tight_layout(); fig.savefig(HERE / "fig_generative_match.png", dpi=150); plt.close(fig)

    print("\n=== generative match (combined partition, switch by reward x run-length) ===")
    print(f"{'cell':>9} {'n':>2} {'corr':>8} {'rmse':>8}")
    for cell in sorted(out, key=lambda c: (c[:2], out[c]["D"])):
        v = out[cell]
        print(f"{cell:>9} {v['n_seeds']:>2} {v['corr_mean']:>8.4f} {v['rmse_mean']:>8.4f}")


if __name__ == "__main__":
    main()
