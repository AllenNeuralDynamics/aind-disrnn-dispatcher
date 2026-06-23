#!/usr/bin/env python
"""Build the v1-vs-v2 data-scaling held-out report: cell-level paired test (loaded),
per-held-out-subject repeated-measures (pulled from the offline heldout-rerun W&B
tables), figures, and FINAL_REPORT.md. Run with the wrapper venv (wandb+scipy+mpl).
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import wandb
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
# offline per-subject re-run groups (v1 + v2, incl. retries)
GROUP_PREFIXES = ("heldout-rerun-v1@", "heldout-rerun-v1-retry@",
                  "heldout-rerun-v2@", "heldout-rerun-v2-retry@")

def _variant(meta):  # 'v1' / 'v2' from meta.variant (strip -retry)
    v = (meta or {}).get("variant", "")
    return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)


def _per_subject_table_dataframe(run):
    for art in run.logged_artifacts():
        if art.type != "run_table":
            continue
        for entry_name in art.manifest.entries:
            if "per_subject_likelihood" not in str(entry_name):
                continue
            return art.get(entry_name).get_dataframe()
    raise ValueError("no heldout/per_subject_likelihood table artifact found")

def collect_per_subject():
    api = wandb.Api()
    rows = []  # variant, ratio, seed, subject, n_trials, ll
    runs = [r for r in api.runs(PROJECT)
            if any((r.group or "").startswith(p) for p in GROUP_PREFIXES) and r.state == "finished"]
    # DEDUP: validation + mass-launch + retries can produce >1 offline run per
    # (variant, ratio, seed). Keep exactly one (the most recent) so seed-averaging
    # isn't skewed by double-counted cells. (The offline finetune is deterministic
    # from the same checkpoint, so duplicates are ~identical, but partial duplication
    # — e.g. only seed 0 re-run — would over-weight that seed.)
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
    runs = list(by_cell.values())
    print(f"  deduped to {len(runs)} offline runs (one per variant,ratio,seed)")
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        var = _variant(meta)
        ratio = meta.get("source_subject_ratio"); seed = meta.get("source_seed")
        if var is None or ratio is None:
            continue
        try:
            df = _per_subject_table_dataframe(r)
        except Exception as e:
            print(f"  [skip {r.name[:40]}] table err: {e}")
            continue
        for _, row in df.iterrows():
            rows.append(dict(variant=var, ratio=round(float(ratio), 3), seed=seed,
                             subject=str(row["heldout_subject_id"]),
                             n_trials=int(row["n_trials"]), ll=float(row["eval_likelihood"])))
    return rows

def per_subject_analysis(rows):
    # average over seeds -> (variant, ratio, subject) mean ll
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["ratio"], r["subject"])].append(r["ll"])
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    ratios = sorted({r["ratio"] for r in rows})
    out = {"per_ratio": {}, "coverage": {}}
    for ratio in ratios:
        subs = sorted({k[2] for k in mean if k[1] == ratio and k[0] == "v1"}
                      & {k[2] for k in mean if k[1] == ratio and k[0] == "v2"})
        d = np.array([mean[("v2", ratio, s)] - mean[("v1", ratio, s)] for s in subs])
        if len(d) >= 5:
            w = stats.wilcoxon(d)
            out["per_ratio"][str(ratio)] = dict(n_subjects=len(d), mean_delta=float(d.mean()),
                                                median_delta=float(np.median(d)),
                                                frac_positive=float((d > 0).mean()),
                                                wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue))
        out["coverage"][str(ratio)] = len(subs)
    return out, mean, ratios

def make_figures(cell, persub, mean, ratios):
    # ratio -> approx D label
    Dlab = {0.016:10, 0.049:30, 0.163:100, 0.489:300, 1.0:614}
    # Fig A: scaling curves v1 vs v2 (cell-level aggregate from paired json)
    pr = cell["per_ratio"]
    xs = sorted(float(k) for k in pr)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([Dlab.get(round(x,3), x) for x in xs], [pr[str(x)]["v1"] for x in xs], "o-", label="v1 (SC off)")
    ax.plot([Dlab.get(round(x,3), x) for x in xs], [pr[str(x)]["v2"] for x in xs], "s-", label="v2 (SC active)")
    ax.set_xscale("log"); ax.set_xlabel("# training mice (D)"); ax.set_ylabel("held-out-mouse likelihood")
    ax.set_title("Held-out generalization vs D"); ax.legend()
    fig.tight_layout(); fig.savefig(HERE/"fig_scaling_v1_v2.png", dpi=150); plt.close(fig)
    # Fig B: per-D delta (v2-v1) with per-subject spread (violin) where available
    fig, ax = plt.subplots(figsize=(6, 4))
    data, labels = [], []
    for ratio in ratios:
        subs = sorted({k[2] for k in mean if k[1]==ratio and k[0]=="v1"} & {k[2] for k in mean if k[1]==ratio and k[0]=="v2"})
        if len(subs) >= 5:
            data.append([mean[("v2",ratio,s)]-mean[("v1",ratio,s)] for s in subs]); labels.append(f"D~{Dlab.get(ratio,ratio)}")
    if data:
        ax.violinplot(data, showmeans=True); ax.set_xticks(range(1,len(labels)+1)); ax.set_xticklabels(labels)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("per-held-out-mouse Δ likelihood (v2−v1)"); ax.set_title("SC effect per held-out mouse, by D")
        fig.tight_layout(); fig.savefig(HERE/"fig_per_subject_delta.png", dpi=150); plt.close(fig)
    return ["fig_scaling_v1_v2.png", "fig_per_subject_delta.png"]

def main():
    cell = json.load(open(HERE/"paired_v1_v2_cell.json"))
    rows = collect_per_subject()
    print(f"collected {len(rows)} per-subject rows")
    persub, mean, ratios = per_subject_analysis(rows)
    figs = make_figures(cell, persub, mean, ratios)
    json.dump({"cell_level": cell, "per_subject": persub}, open(HERE/"report_data.json","w"), indent=2)
    # summary print
    print("\n=== per-subject repeated-measures (v2-v1, paired by held-out mouse, avg over seeds) ===")
    for r, s in persub["per_ratio"].items():
        print(f"  ratio={r}: n={s['n_subjects']} meanΔ={s['mean_delta']:+.5f} medΔ={s['median_delta']:+.5f} "
              f"frac+={s['frac_positive']:.2f} Wilcoxon p={s['wilcoxon_p']:.2e}")
    return cell, persub, figs

if __name__ == "__main__":
    main()
