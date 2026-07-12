#!/usr/bin/env python
"""Mature few-shot adaptation curve: does the naive K=1 overfit crash survive on
MATURE-only held-out sessions?

All-stage few-shot (analysis/fewshot_curve.json) shows a K=1 crater: adapting the
subject embedding on a *single* held-out session drops LL well below zero-shot (K0),
recovering by K=4. Hypothesis (FUTURE_DIRECTIONS §8): the crash is caused by adapting
on a naive *early-stage* session. If so, restricting eval to MATURE sessions
(STAGE_FINAL/GRADUATED) should remove the crash (K1 >= K0).

Pulls the 6 g6e mature groups (zeroshotM=K0, k1M=K1, k4M=K4 for v1/v2), reads each run's
heldout/per_subject_likelihood table, cohort-means eval_likelihood per (variant, D, K),
and compares the K=1 dip mature-only vs all-stage. Run with the wrapper venv.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _meta import build_meta

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
# K-level -> group prefixes (variant folded in). Match by prefix; dedup by created_at
# keeps the newest run per cell, so the latest g6e launch wins over any stopped/older
# same-named launch.
KLEVELS = {
    0: ("heldout-zeroshot-v1-zeroshotM@", "heldout-zeroshot-v2-zeroshotM@"),
    1: ("heldout-rerun-v1-k1M@", "heldout-rerun-v2-k1M@"),
    4: ("heldout-rerun-v1-k4M@", "heldout-rerun-v2-k4M@"),
}
DLAB = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}


def _variant(meta):
    v = (meta or {}).get("variant", "")
    return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)


def _per_subject_df(run):
    for art in run.logged_artifacts():
        if art.type != "run_table":
            continue
        for entry in art.manifest.entries:
            if "per_subject_likelihood" in str(entry):
                return art.get(entry).get_dataframe()
    raise ValueError("no per_subject_likelihood table")


def collect():
    api = wandb.Api()
    # rows: (variant, D, K, seed, subject, ll)
    rows = []
    groups_seen = set()
    for K, prefixes in KLEVELS.items():
        runs = [r for r in api.runs(PROJECT)
                if any((r.group or "").startswith(p) for p in prefixes)
                and r.state == "finished"]
        groups_seen.update(r.group for r in runs if r.group)
        # dedup to one run per (variant, ratio, seed) — newest wins
        by_cell = {}
        for r in runs:
            meta = r.config.get("meta", {}) or {}
            var = _variant(meta)
            ratio = meta.get("source_subject_ratio")
            seed = meta.get("source_seed")
            if var is None or ratio is None:
                continue
            key = (var, round(float(ratio), 3), seed)
            prev = by_cell.get(key)
            if prev is None or str(getattr(r, "created_at", "")) > str(getattr(prev, "created_at", "")):
                by_cell[key] = r
        print(f"  K={K}: {len(runs)} finished runs -> {len(by_cell)} cells")
        for (var, ratio, seed), r in by_cell.items():
            try:
                df = _per_subject_df(r)
            except Exception as e:
                print(f"    [skip {r.name[:40]}] {e}")
                continue
            for _, row in df.iterrows():
                rows.append(dict(variant=var, D=DLAB.get(ratio, ratio), ratio=ratio,
                                 K=K, seed=seed, subject=str(row["heldout_subject_id"]),
                                 ll=float(row["eval_likelihood"])))
    return rows, sorted(groups_seen)


def cohort_means(rows):
    # mean over seeds per (variant,D,K,subject), then over subjects -> cohort mean
    persub = defaultdict(list)
    for r in rows:
        persub[(r["variant"], r["D"], r["K"], r["subject"])].append(r["ll"])
    submean = {k: float(np.mean(v)) for k, v in persub.items()}
    cohort = defaultdict(list)
    ncov = defaultdict(set)
    for (var, D, K, sub), m in submean.items():
        cohort[(var, D, K)].append(m)
        ncov[(var, D, K)].add(sub)
    out = {}
    for key, vals in cohort.items():
        out[key] = dict(ll=float(np.mean(vals)), n=len(ncov[key]))
    return out


def main():
    rows, groups = collect()
    print(f"collected {len(rows)} per-subject rows")
    cohort = cohort_means(rows)
    Ds = sorted({k[1] for k in cohort})
    variants = ["v1", "v2"]
    Ks = [0, 1, 4]

    # all-stage reference: fewshot_curve.json arrays are [K0, K1, K4, Kfull]
    allstage = json.load(open(HERE / "fewshot_curve.json"))
    idx = {0: 0, 1: 1, 4: 2, "full": 3}

    # graft the fully-adapted (K=full) MATURE point from mature_sc_verdict.json
    # (heldout-rerun-*-mature groups): v1_mat given directly, v2_mat = v1_mat + Δ(v2-v1).
    verdict = json.load(open(HERE / "mature_sc_verdict.json"))
    full_mat = {}  # (variant, D) -> ll
    for Dk, v in verdict.items():
        full_mat[("v1", int(Dk))] = v["v1_mat"]
        full_mat[("v2", int(Dk))] = v["v1_mat"] + v["mature_delta"]

    result = {
        "_meta": build_meta("analysis/mature_fewshot_curve.py", groups),
        "mature": {},
        "k1_dip_vs_k0": {},
    }
    for var in variants:
        for D in Ds:
            cell = {}
            for K in Ks:
                c = cohort.get((var, D, K))
                if c:
                    cell[str(K)] = c
            if (var, D) in full_mat:
                cell["full"] = {"ll": full_mat[(var, D)], "n": 117}
            result["mature"][f"{var}_D{D}"] = cell
            # K=1 dip relative to K=0, mature vs all-stage
            mk0 = cohort.get((var, D, 0)); mk1 = cohort.get((var, D, 1))
            ak = allstage.get(f"{var}_D{D}")
            entry = {}
            if mk0 and mk1:
                entry["mature_k1_minus_k0"] = mk1["ll"] - mk0["ll"]
            if ak:
                entry["allstage_k1_minus_k0"] = ak[idx[1]] - ak[idx[0]]
            result["k1_dip_vs_k0"][f"{var}_D{D}"] = entry

    json.dump(result, open(HERE / "mature_fewshot_curve.json", "w"), indent=2)

    # figure: K-curve per D, mature (solid) vs all-stage (dashed), v1 & v2 panels.
    # x positions: K0,K1,K4 evenly, then Kfull at the right end (categorical).
    Kpts = [0, 1, 4, "full"]
    xpos = [0, 1, 2, 3]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, var in zip(axes, variants):
        for i, D in enumerate(Ds):
            color = cmap(i / max(1, len(Ds) - 1))
            mat = [(cohort.get((var, D, K), {}).get("ll") if K != "full"
                    else full_mat.get((var, D))) for K in Kpts]
            ax.plot(xpos, mat, "o-", color=color, label=f"D~{D} mature")
            ak = allstage.get(f"{var}_D{D}")
            if ak:
                ax.plot(xpos, [ak[idx[K]] for K in Kpts], "s--", color=color, alpha=0.5)
        ax.set_xticks(xpos); ax.set_xticklabels(["0", "1", "4", "full"])
        ax.set_xlabel("K (adaptation sessions)")
        ax.set_title(f"{var} — solid=mature, dashed=all-stage")
    axes[0].set_ylabel("held-out-mouse likelihood")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Few-shot adaptation: does the K=1 crash survive mature-only?")
    fig.tight_layout()
    fig.savefig(HERE / "fig_mature_fewshot_curve.png", dpi=150)
    plt.close(fig)

    # summary print
    print("\n=== K=1 dip (LL[K1]-LL[K0]); negative = crash ===")
    print(f"{'cell':>10} {'mature':>12} {'all-stage':>12}")
    for cell, e in result["k1_dip_vs_k0"].items():
        m = e.get("mature_k1_minus_k0"); a = e.get("allstage_k1_minus_k0")
        print(f"{cell:>10} {('%+.5f'%m) if m is not None else '   n/a':>12} "
              f"{('%+.5f'%a) if a is not None else '   n/a':>12}")


if __name__ == "__main__":
    main()
