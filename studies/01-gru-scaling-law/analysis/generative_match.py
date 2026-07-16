#!/usr/bin/env python
"""Generative behavioral-match vs D (2nd-order validation).

Rolls-out GRU as an agent; compares two model-vs-animal behavioral curves on the COMBINED
session partition:

  (1) switch-triggered: p_switch(t+1) conditioned on (reward at t) x (preceding run-length),
      i.e. ``post_switch_by_reward_and_run_length`` -- 4 bins, the corr~0.96 headline.
  (2) history-dependent: p_switch(t) conditioned on the previous N trials' (choice, reward)
      pattern, for N in {1, 2, 3}, in two encodings (``abstract`` canonicalises the first
      trial to A; ``detailed`` keeps L/R identity). N=3 abstract = 32 pattern bins.

For each (variant, D) cell we average corr / RMSE over 3 seeds. RMSE is sqrt(MSE) of the
subject-balanced delta MSE under ``delta_significance_summary`` (per-subject mean delta,
then MSE across subjects).

Run with the wrapper venv (or any env with wandb + numpy + matplotlib). Source W&B groups:
``generative-v{1,2}@<launch_id>`` in project ``AIND-disRNN/mice_data_scaling``.
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

from _meta import build_meta
from wandb_keys import (SWITCH_CORR_OVERALL, hist_corr, hist_mse, switch_corr,
                        switch_mse)

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
PREFIXES = ("generative-v1@", "generative-v2@")
DLAB = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}

# RL baselines: rolled out generatively under study 05 (variants/generative-rl-baseline), same
# task construction as this study's own GRU rollouts (build_curriculum_matched_task). Read
# cross-study rather than duplicated -- that variant's notes.md has the full methodology/
# provenance. Single point each at D=614 (per-subject fits over all 614 mice), NOT a D-sweep.
RL_JSON_DIR = (
    HERE.parent.parent / "05-disrnn-scaling-law" / "variants" / "generative-rl-baseline"
    / "rl_rollout_summaries"
)
RL_LABELS = {"ctt": "compare-to-threshold", "bari": "Bari", "hattori": "Hattori"}

# (1) switch-triggered: run-length-resolved switch curve (the corr~0.96 headline).
STAT = "post_switch_by_reward_and_run_length"
K_CORR = switch_corr(STAT)
K_MSE = switch_mse(STAT)
K_CORR_OVERALL = SWITCH_CORR_OVERALL

# (2) history-dependent: p_switch | last N trials' (choice, reward) pattern.
HIST_PATTERNS = ("abstract", "detailed")
HIST_N_BACKS = (1, 2, 3)
HIST_HEADLINE = ("abstract", 3)  # (pattern_type, n_back) shown in fig_generative_match_history.png


def _variant(meta):
    v = (meta or {}).get("variant", "")
    return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)


def collect():
    api = wandb.Api()
    runs = [r for r in api.runs(PROJECT)
            if any((r.group or "").startswith(p) for p in PREFIXES) and r.state == "finished"]
    groups = sorted({r.group for r in runs if r.group})
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
        row = dict(variant=var, ratio=ratio, D=DLAB.get(ratio, ratio), seed=seed,
                   corr=float(corr), corr_overall=float(corr_o) if corr_o is not None else None,
                   rmse=math.sqrt(float(mse)))
        # history-dependent metric: corr + sqrt(subject-balanced MSE), per (pattern, n_back).
        for pattern in HIST_PATTERNS:
            for n in HIST_N_BACKS:
                hc = s.get(hist_corr(pattern, n))
                hm = s.get(hist_mse(pattern, n))
                row[f"hist_{pattern}_n{n}_corr"] = float(hc) if hc is not None else None
                row[f"hist_{pattern}_n{n}_rmse"] = math.sqrt(float(hm)) if hm is not None else None
        rows.append(row)
    # All finished cells missing the required scalars means a renamed key, not
    # a few partial runs -> fail loudly rather than emit an empty report.
    if by_cell and not rows:
        raise KeyError(
            f"none of {len(by_cell)} finished generative cells carry {K_CORR!r}/"
            f"{K_MSE!r} (wrapper schema changed? see analysis/wandb_keys.py)"
        )
    return rows, groups


def rl_reference():
    """RL baselines' generative match at D=614 (study 05's rollout, same task construction).

    quantitative_summary.json per alias has the SAME nested structure the wrapper logs to W&B
    (see wandb_keys.py) -- just not flattened into dotted keys, since these standalone rollouts
    were never logged to a W&B run.
    """
    out = {}
    for alias, label in RL_LABELS.items():
        path = RL_JSON_DIR / f"{alias}_quantitative_summary.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        sw = d["switch_triggered"]["quantitative_summary"]["subject_mean"][STAT]
        sw_mse = d["switch_triggered"]["delta_significance_summary"][STAT][
            "subject_balanced_error_summary"
        ]["mean_squared_error"]
        hp, hn = HIST_HEADLINE
        hist = d["history_dependent"]["quantitative_summary"]["subject_mean"][hp][str(hn)]
        hist_mse = d["history_dependent"]["delta_significance_summary"][hp][str(hn)][
            "subject_balanced_error_summary"
        ]["mean_squared_error"]
        out[alias] = dict(
            label=label,
            D=614,
            corr=float(sw["correlation"]),
            rmse=math.sqrt(float(sw_mse)),
            hist_corr=float(hist["correlation"]),
            hist_rmse=math.sqrt(float(hist_mse)),
        )
    return out


def main():
    rows, groups = collect()
    print(f"collected {len(rows)} generative cells")
    rl = rl_reference()
    # (1) switch-triggered: aggregate over seeds per (variant, D).
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["variant"], r["D"])]["corr"].append(r["corr"])
        agg[(r["variant"], r["D"])]["rmse"].append(r["rmse"])
    out = {}
    for (var, D), m in agg.items():
        out[f"{var}_D{D}"] = dict(D=D, n_seeds=len(m["corr"]),
                                  corr_mean=float(np.mean(m["corr"])), corr_sd=float(np.std(m["corr"])),
                                  rmse_mean=float(np.mean(m["rmse"])), rmse_sd=float(np.std(m["rmse"])))

    # (2) history-dependent: same aggregation, nested by pattern_type / n_back.
    hist_out: dict = {pattern: {str(n): {} for n in HIST_N_BACKS} for pattern in HIST_PATTERNS}
    for pattern in HIST_PATTERNS:
        for n in HIST_N_BACKS:
            cell_agg = defaultdict(lambda: defaultdict(list))
            for r in rows:
                c = r.get(f"hist_{pattern}_n{n}_corr"); e = r.get(f"hist_{pattern}_n{n}_rmse")
                if c is None or e is None:
                    continue
                cell_agg[(r["variant"], r["D"])]["corr"].append(c)
                cell_agg[(r["variant"], r["D"])]["rmse"].append(e)
            for (var, D), m in cell_agg.items():
                hist_out[pattern][str(n)][f"{var}_D{D}"] = dict(
                    D=D, n_seeds=len(m["corr"]),
                    corr_mean=float(np.mean(m["corr"])), corr_sd=float(np.std(m["corr"])),
                    rmse_mean=float(np.mean(m["rmse"])), rmse_sd=float(np.std(m["rmse"])),
                )

    json.dump({"_meta": build_meta("analysis/generative_match.py", groups),
               **out, "history_dependent": hist_out, "rl_reference": rl},
              open(HERE / "generative_match.json", "w"), indent=2)

    # --- figure 1: switch-triggered match vs D (unchanged) ---------------------
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
    rl_markers = {"ctt": "^", "bari": "v", "hattori": "D"}
    for alias, v in rl.items():
        axc.scatter([v["D"]], [v["corr"]], marker=rl_markers.get(alias, "x"), s=70,
                    color="#c44e52", zorder=5, label=f"RL: {v['label']}")
        axr.scatter([v["D"]], [v["rmse"]], marker=rl_markers.get(alias, "x"), s=70,
                    color="#c44e52", zorder=5, label=f"RL: {v['label']}")
    for ax in (axc, axr):
        # upper-left is the corner every series (GRU v1/v2, RL) leaves clear at every D.
        ax.set_xscale("log"); ax.set_xlabel("# training mice (D)"); ax.legend(loc="upper left", fontsize=8)
    axc.set_ylabel("subject-mean correlation (model vs animal)")
    axc.set_title("Generative switch-curve match vs D")
    axr.set_ylabel("subject-balanced RMSE")
    axr.set_title("Generative switch-curve error vs D")
    fig.tight_layout(); fig.savefig(HERE / "fig_generative_match.png", dpi=150); plt.close(fig)

    # --- figure 2: history-dependent match vs D (abstract n=3 headline) --------
    hp, hn = HIST_HEADLINE
    cells = hist_out[hp][str(hn)]
    Ds_h = sorted({c["D"] for c in cells.values()})
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(11, 4.3))
    for var, mk in (("v1", "o-"), ("v2", "s-")):
        xs = [D for D in Ds_h if f"{var}_D{D}" in cells]
        cm = [cells[f"{var}_D{D}"]["corr_mean"] for D in xs]
        cs = [cells[f"{var}_D{D}"]["corr_sd"] for D in xs]
        rm = [cells[f"{var}_D{D}"]["rmse_mean"] for D in xs]
        rs = [cells[f"{var}_D{D}"]["rmse_sd"] for D in xs]
        lab = "v1 (SC off)" if var == "v1" else "v2 (SC active)"
        axc.errorbar(xs, cm, yerr=cs, fmt=mk, capsize=3, label=lab)
        axr.errorbar(xs, rm, yerr=rs, fmt=mk, capsize=3, label=lab)
    for alias, v in rl.items():
        axc.scatter([v["D"]], [v["hist_corr"]], marker=rl_markers.get(alias, "x"), s=70,
                    color="#c44e52", zorder=5, label=f"RL: {v['label']}")
        axr.scatter([v["D"]], [v["hist_rmse"]], marker=rl_markers.get(alias, "x"), s=70,
                    color="#c44e52", zorder=5, label=f"RL: {v['label']}")
    for ax in (axc, axr):
        ax.set_xscale("log"); ax.set_xlabel("# training mice (D)"); ax.legend(loc="upper left", fontsize=8)
    axc.set_ylabel(f"subject-mean correlation ({hp} n={hn})")
    axc.set_title(f"Generative 3-trial-back history match vs D ({hp})")
    axr.set_ylabel(f"subject-balanced RMSE ({hp} n={hn})")
    axr.set_title(f"Generative 3-trial-back history error vs D ({hp})")
    fig.tight_layout(); fig.savefig(HERE / "fig_generative_match_history.png", dpi=150); plt.close(fig)

    # --- console summary --------------------------------------------------------
    print("\n=== switch-triggered (combined, post-switch by reward x run-length, 4 bins) ===")
    print(f"{'cell':>9} {'n':>2} {'corr':>8} {'rmse':>8}")
    for cell in sorted(out, key=lambda c: (c[:2], out[c]["D"])):
        v = out[cell]
        print(f"{cell:>9} {v['n_seeds']:>2} {v['corr_mean']:>8.4f} {v['rmse_mean']:>8.4f}")
    for pattern in HIST_PATTERNS:
        for n in HIST_N_BACKS:
            section = hist_out[pattern][str(n)]
            if not section:
                continue
            print(f"\n=== history-dependent {pattern} n_back={n} ===")
            print(f"{'cell':>9} {'n':>2} {'corr':>8} {'rmse':>8}")
            for cell in sorted(section, key=lambda c: (c[:2], section[c]["D"])):
                v = section[cell]
                print(f"{cell:>9} {v['n_seeds']:>2} {v['corr_mean']:>8.4f} {v['rmse_mean']:>8.4f}")

    if rl:
        print(f"\n=== RL baselines at D=614 (switch corr/rmse, {hp} n={hn} history corr/rmse) ===")
        for alias, v in rl.items():
            print(f"{v['label']:>22}  switch {v['corr']:.4f}/{v['rmse']:.4f}   "
                  f"history {v['hist_corr']:.4f}/{v['hist_rmse']:.4f}")


if __name__ == "__main__":
    main()
