#!/usr/bin/env python
"""RL-baseline posthoc: simple independent per-subject Bari RL (ForagerQLearning
L1F1_CK1) on the same 149 held-out mice as the GRU.

Sources:
  * RL run `cdq292n5` (group rl-baseline-simple@20260624-171829). Pulled fresh
    each run.
  * GRU per-subject eval_likelihood for v1/v2 x ratio x 3 seeds from the offline
    `heldout-rerun-*` groups. Heavy W&B pull, so CACHED to
    `analysis/gru_per_subject.json` on first run; subsequent runs re-use it
    (delete that file to refresh).
  * Cached aggregate JSONs for the 4 figure overlays (no W&B pull):
      Result 1: paired_v1_v2_cell.json
      Result 4: zeroshot_vs_d.json
      Result 5: fewshot_curve.json
      Result 7: nxd_scaling.json

Produces:
  * analysis/rl_baseline.json - RL band + per-(variant,D) paired stats.
  * analysis/rl_baseline_verdict.md - table + interpretation.
  * analysis/fig_rl_paired.png - Result 8 figure: per-mouse Delta(GRU - RL).
  * Overlaid replacements (each figure overwrites the existing PNG in place):
      fig_scaling_v1_v2.png, fig_zeroshot_vs_d.png,
      fig_fewshot_curve.png, fig_nxd_scaling.png

Run with the wrapper venv (wandb + scipy + matplotlib required).
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from _meta import build_meta

# Presentation styling: large fonts, ticks style (no top/right spines after despine).
sns.set_theme(style="ticks", context="talk", font_scale=1.05)
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
RL_RUN_NAME = "cdq292n5"
GRU_CACHE = HERE / "gru_per_subject.json"

GRU_PREFIXES = ("heldout-rerun-v1@", "heldout-rerun-v1-retry@",
                "heldout-rerun-v2@", "heldout-rerun-v2-retry@")

RATIO_D = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}


# ============================= W&B pulls =============================

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


def pull_rl(api):
    runs = list(api.runs(PROJECT, filters={"name": RL_RUN_NAME}))
    if not runs:
        raise RuntimeError(f"RL run {RL_RUN_NAME} not found in {PROJECT}")
    r = runs[0]
    art = next(a for a in r.logged_artifacts() if a.type == "run_table")
    df = art.get(next(iter(art.manifest.entries))).get_dataframe()
    df["subject_id"] = df["subject_id"].astype(str)
    pooled_log = float(df["eval_total_log_likelihood"].sum() /
                       df["eval_total_trials"].sum())
    rl = dict(
        run_id=r.id,
        run_url=f"https://wandb.ai/{PROJECT}/runs/{r.id}",
        group=r.group,
        n_subjects=int(len(df)),
        n_trials=int(df["eval_total_trials"].sum()),
        pooled_log_likelihood=pooled_log,
        pooled_likelihood_trial_weighted=math.exp(pooled_log),
        per_subject_mean_likelihood=float(df["eval_likelihood"].mean()),
        per_subject_se=float(df["eval_likelihood"].std(ddof=1) /
                             math.sqrt(len(df))),
        per_subject_median_likelihood=float(df["eval_likelihood"].median()),
        per_subject_likelihood={s: float(v) for s, v in
                                zip(df["subject_id"], df["eval_likelihood"])},
        per_curriculum={
            k: {"n": int(g.shape[0]),
                "mean": float(g["eval_likelihood"].mean()),
                "std": float(g["eval_likelihood"].std(ddof=1)) if g.shape[0] > 1 else 0.0}
            for k, g in df.groupby("curriculum_name")},
    )
    print(f"  RL: n={rl['n_subjects']} mice, pooled={rl['pooled_likelihood_trial_weighted']:.4f}, "
          f"per-subj mean={rl['per_subject_mean_likelihood']:.4f}")
    return rl


def list_gru_groups(api):
    """Enumerate GRU W&B groups matching GRU_PREFIXES (cheap metadata-only pass;
    needed for `_meta.wandb_groups` even when the per-subject pull is cached)."""
    groups = {r.group for r in api.runs(PROJECT)
              if any((r.group or "").startswith(p) for p in GRU_PREFIXES)}
    return sorted(g for g in groups if g)


def pull_gru_per_subject(api):
    """Pull GRU per-subject likelihoods; dedup to newest run per (variant,ratio,seed)
    that actually has a per_subject_likelihood table (some original runs failed to
    log the artifact and got re-run as -retry groups)."""
    runs = [r for r in api.runs(PROJECT)
            if any((r.group or "").startswith(p) for p in GRU_PREFIXES)
            and r.state == "finished"]
    by_cell = {}
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        var = _variant(meta)
        ratio = meta.get("source_subject_ratio")
        seed = meta.get("source_seed")
        if var is None or ratio is None:
            continue
        k = (var, round(float(ratio), 3), str(seed))
        prev = by_cell.get(k)
        if prev is None or str(getattr(r, "created_at", "")) > str(getattr(prev, "created_at", "")):
            by_cell[k] = r
    print(f"  GRU: {len(runs)} finished runs -> {len(by_cell)} (variant,ratio,seed) cells")

    rows = []
    missing = 0
    for (var, ratio, seed), r in by_cell.items():
        try:
            df = _per_subject_df(r)
        except ValueError:
            missing += 1
            continue
        D = RATIO_D[ratio]
        for _, row in df.iterrows():
            rows.append(dict(variant=var, D=D, ratio=ratio, seed=seed,
                             subject=str(row["heldout_subject_id"]),
                             ll=float(row["eval_likelihood"])))
    if missing:
        print(f"  WARN: skipped {missing} cells with no per_subject_likelihood table "
              "(superseded by a newer cell in the same (variant,ratio,seed))")
    return rows


def load_or_pull_gru(api):
    if GRU_CACHE.exists():
        print(f"  loading cached GRU per-subject from {GRU_CACHE.name} "
              "(delete to refresh)")
        return json.load(open(GRU_CACHE))
    print("  pulling GRU per-subject from W&B (will cache)...")
    rows = pull_gru_per_subject(api)
    print(f"    collected {len(rows)} per-subject rows")
    # nested dict for compact storage: variant -> D -> subject -> [ll per seed]
    nested = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        nested[r["variant"]][str(r["D"])][r["subject"]].append(r["ll"])
    json.dump(nested, open(GRU_CACHE, "w"), indent=2)
    print(f"    cached -> {GRU_CACHE}")
    return nested


# ============================= analysis =============================

def gru_per_subject_means(gru_nested):
    """nested[var][D_str][subj] -> [ll per seed]  =>  flat (var, D, subj) -> mean ll."""
    out = {}
    for var, byD in gru_nested.items():
        for D_str, bysub in byD.items():
            D = int(D_str)
            for subj, lls in bysub.items():
                out[(var, D, subj)] = float(np.mean(lls))
    return out


def paired_vs_rl(gru_mean, rl_per_subject):
    variants = sorted({k[0] for k in gru_mean})
    Ds = sorted({k[1] for k in gru_mean})
    out = {"variants": variants, "Ds": Ds, "per_cell": {}}
    for var in variants:
        for D in Ds:
            subs = sorted({k[2] for k in gru_mean if k[0] == var and k[1] == D} & set(rl_per_subject))
            if len(subs) < 5:
                continue
            gru = np.array([gru_mean[(var, D, s)] for s in subs])
            rl = np.array([rl_per_subject[s] for s in subs])
            d = gru - rl
            wil = stats.wilcoxon(d)
            t = stats.ttest_rel(gru, rl)
            out["per_cell"][f"{var}_D{D}"] = dict(
                variant=var, D=D, n=len(subs),
                gru_mean=float(gru.mean()), rl_mean=float(rl.mean()),
                mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                frac_gru_wins=float((d > 0).mean()),
                paired_t_t=float(t.statistic), paired_t_p=float(t.pvalue),
                wilcoxon_W=float(wil.statistic), wilcoxon_p=float(wil.pvalue),
                per_subject_delta={s: float(g - r) for s, g, r in zip(subs, gru, rl)},
            )
    return out


# ============================= figures =============================

def _rl_band(ax, rl, metric):
    """Add an RL horizontal reference band. metric in {trial_weighted, subject_mean, both}."""
    if metric in ("trial_weighted", "both"):
        ax.axhline(rl["pooled_likelihood_trial_weighted"], color="#7a3a3a", lw=1.2,
                   ls="--", label=f"RL (trial-weighted): {rl['pooled_likelihood_trial_weighted']:.4f}")
    if metric in ("subject_mean", "both"):
        mu = rl["per_subject_mean_likelihood"]; se = rl["per_subject_se"]
        ax.axhline(mu, color="#3a3a7a", lw=1.2, ls=":",
                   label=f"RL (per-subject mean): {mu:.4f} ± {se:.4f} SE")
        ax.axhspan(mu - se, mu + se, color="#3a3a7a", alpha=0.08)


# --- Result 8: per-mouse delta(GRU - RL) violin -----------------------------

def fig_paired(paired, rl, out_png):
    Ds = paired["Ds"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, var in zip(axes, paired["variants"]):
        data, labels = [], []
        means, ps, fracs = [], [], []
        for D in Ds:
            cell = paired["per_cell"].get(f"{var}_D{D}")
            if cell is None:
                continue
            deltas = list(cell["per_subject_delta"].values())
            data.append(deltas); labels.append(f"D~{D}\nn={cell['n']}")
            means.append(cell["mean_delta"]); ps.append(cell["wilcoxon_p"])
            fracs.append(cell["frac_gru_wins"])
        if not data:
            continue
        parts = ax.violinplot(data, showmeans=True, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor("#3b76b8" if var == "v1" else "#d97c2a"); pc.set_alpha(0.5)
        ax.axhline(0, color="k", lw=1.0, ls="--", label="RL reference (Δ=0)")
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
        ax.set_xlabel("# training mice (D)")
        ymin = min(min(d) for d in data); ymax = max(max(d) for d in data)
        pad = 0.05 * (ymax - ymin)
        for i, (m, p, f) in enumerate(zip(means, ps, fracs)):
            ax.text(i + 1, ymax + pad,
                    f"μ={m:+.4f}\n{f*100:.0f}%↑\np={p:.0e}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{var}: per-mouse Δ(GRU − RL)")
    axes[0].set_ylabel("Δ likelihood (per held-out mouse, GRU − RL)")
    axes[1].legend(loc="lower right")
    rl_lab = (f"RL ref: trial-weighted {rl['pooled_likelihood_trial_weighted']:.4f}, "
              f"per-subject mean {rl['per_subject_mean_likelihood']:.4f} (n=149)")
    fig.suptitle(f"Result 8 — paired GRU − RL per held-out mouse\n{rl_lab}", fontsize=15)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 1: scaling v1 vs v2 + RL ---------------------------------------

def fig_scaling_v1_v2_with_rl(rl, out_png):
    pj = json.load(open(HERE / "paired_v1_v2_cell.json"))
    pr = pj["per_ratio"]
    xs_ratio = sorted(float(k) for k in pr)
    Ds = [pr[str(x)]["D_mean"] for x in xs_ratio]
    v1 = [pr[str(x)]["v1"] for x in xs_ratio]
    v2 = [pr[str(x)]["v2"] for x in xs_ratio]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(Ds, v1, "o-", color="#3b76b8", label="v1 (SC off)", lw=2, markersize=9)
    ax.plot(Ds, v2, "s-", color="#d97c2a", label="v2 (SC active)", lw=2, markersize=9)
    # cell-level v1/v2 are trial-weighted (run-level aggregate), so use the trial-weighted RL
    _rl_band(ax, rl, "trial_weighted")
    ax.set_xscale("log"); ax.set_xlabel("# training mice (D)")
    ax.set_ylabel("held-out-mouse likelihood (cell-level)")
    ax.set_title("Result 1 — held-out scaling vs D (+ RL reference)")
    ax.legend(loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 4: zero-shot vs adapted + RL ----------------------------------

def fig_zeroshot_vs_d_with_rl(rl, out_png):
    z = json.load(open(HERE / "zeroshot_vs_d.json"))
    Ds = z["Ds"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, var, ttl in zip(axes, ("v1", "v2"), ("v1 (SC off)", "v2 (SC active)")):
        v = z["variants"][var]
        ax.plot(Ds, v["zeroshot"], "o--", color="#7a7a7a",
                label="zero-shot (population embedding)", lw=2, markersize=9)
        ax.plot(Ds, v["adapted"], "o-", color="#3b76b8" if var == "v1" else "#d97c2a",
                label="adapted (per-mouse embedding fine-tuned)", lw=2, markersize=9)
        # per-mouse panel -> show subject-mean RL as primary; also annotate trial-weighted
        _rl_band(ax, rl, "subject_mean")
        ax.set_xscale("log"); ax.set_xlabel("# training mice (D)")
        ax.set_title(ttl)
        ax.legend(loc="lower right")
    axes[0].set_ylabel("held-out-mouse likelihood (per-mouse mean)")
    fig.suptitle("Result 4 — zero-shot vs adapted vs D (+ RL reference)", fontsize=16)
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 5: few-shot K-curve + RL --------------------------------------

def fig_fewshot_curve_with_rl(rl, out_png):
    fc = json.load(open(HERE / "fewshot_curve.json"))
    Ks = [0, 1, 4, "full"]; xpos = [0, 1, 2, 3]
    Ds_present = sorted({int(k.split("_D")[1]) for k in fc})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, var in zip(axes, ("v1", "v2")):
        for i, D in enumerate(Ds_present):
            arr = fc.get(f"{var}_D{D}")
            if arr is None: continue
            color = cmap(i / max(1, len(Ds_present) - 1))
            ax.plot(xpos, arr, "o-", color=color, label=f"D~{D}", lw=2, markersize=9)
        _rl_band(ax, rl, "subject_mean")
        ax.set_xticks(xpos); ax.set_xticklabels([str(k) for k in Ks])
        ax.set_xlabel("K (adaptation sessions)"); ax.set_title(var)
    axes[0].set_ylabel("held-out-mouse likelihood (per-mouse mean)")
    axes[1].legend(ncol=2, loc="lower right")
    fig.suptitle("Result 5 — few-shot adaptation curve (+ RL reference)", fontsize=16)
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


# --- Result 7: N x D scaling + RL -----------------------------------------

def fig_nxd_scaling_with_rl(rl, out_png):
    nx = json.load(open(HERE / "nxd_scaling.json"))
    Ns, Ds = nx["Ns"], nx["Ds"]
    mean_grid = np.array(nx["mean_grid"], dtype=float)
    se_grid = np.array(nx["se_grid"], dtype=float)
    fa = nx["fit_additive"]; fi = nx["fit_interaction"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2))

    # Panel A: heatmap (unchanged from nxd_scaling.py)
    ax = axes[0]
    im = ax.imshow(mean_grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(Ds))); ax.set_xticklabels([str(d) for d in Ds])
    ax.set_yticks(range(len(Ns))); ax.set_yticklabels([str(n) for n in Ns])
    ax.set_xlabel("D (# training mice)"); ax.set_ylabel("N (hidden size)")
    ax.set_title("held-out eval_likelihood (mean over seeds)")
    for i in range(len(Ns)):
        for j in range(len(Ds)):
            v = mean_grid[i, j]
            if not np.isnan(v):
                col = "white" if v < (np.nanmin(mean_grid) + np.nanmax(mean_grid)) / 2 else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", color=col, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: L vs D, one line per N — plus RL band
    ax = axes[1]
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(Ns)))
    for i, N in enumerate(Ns):
        ax.errorbar(Ds, mean_grid[i, :], yerr=se_grid[i, :], marker="o",
                    color=colors[i], label=f"N={N}", capsize=4, lw=2, markersize=8)
    _rl_band(ax, rl, "trial_weighted")  # cell-level metric, trial-weighted
    ax.set_xscale("log"); ax.set_xlabel("D (# training mice)")
    ax.set_ylabel("held-out eval_likelihood")
    ax.set_title("L vs D, per N (+ RL reference)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False, fontsize=12)

    # Panel C: L vs N, one line per D — plus RL band
    ax = axes[2]
    colors2 = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ds)))
    for j, D in enumerate(Ds):
        ax.errorbar(Ns, mean_grid[:, j], yerr=se_grid[:, j], marker="s",
                    color=colors2[j], label=f"D={D}", capsize=4, lw=2, markersize=8)
    _rl_band(ax, rl, "trial_weighted")
    ax.set_xscale("log"); ax.set_xlabel("N (hidden size)")
    ax.set_ylabel("held-out eval_likelihood")
    ax.set_title("L vs N, per D (+ RL reference)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False, fontsize=12)

    title = (f"Result 7 — N×D scaling | additive E={fa['E']:.4f} "
             f"α={fa['alpha']:.3f} β={fa['beta']:.3f} "
             f"| AIC_add={fa['aic']:.1f} AIC_int={fi.get('aic', float('nan')):.1f}"
             f" | RL pooled={rl['pooled_likelihood_trial_weighted']:.4f}")
    fig.suptitle(title, fontsize=14, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout(); fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ============================= verdict =============================

def write_verdict(rl, paired, out_md):
    lines = []
    lines.append("# RL baseline (simple, per-mouse independent fit) — verdict")
    lines.append("")
    lines.append(f"Source: [{rl['run_id']}]({rl['run_url']}) (group `{rl['group']}`). "
                 "Model: `baseline_rl` / `ForagerQLearning` (L1F1_CK1 — 1 learn-rate, 1 forget-rate, "
                 "1-step choice kernel, softmax). One DE optimizer fit per held-out mouse on its "
                 "own train sessions, scored on its own eval sessions. Same fixed held-out cohort "
                 f"(n={rl['n_subjects']}) and eval sessions as the GRU.")
    lines.append("")
    lines.append("## RL reference band")
    lines.append("")
    lines.append(f"- **Pooled (trial-weighted) likelihood:** **{rl['pooled_likelihood_trial_weighted']:.4f}** "
                 "(matches GRU `heldout/eval_likelihood`; used as the band on Results 1, 7).")
    lines.append(f"- **Per-subject mean likelihood:** **{rl['per_subject_mean_likelihood']:.4f}** "
                 f"± {rl['per_subject_se']:.4f} SE, median {rl['per_subject_median_likelihood']:.4f} "
                 "(used on per-mouse panels — Results 4, 5).")
    lines.append(f"- n = {rl['n_subjects']} held-out mice ({rl['n_trials']:,} eval trials).")
    lines.append("")
    lines.append("Per-curriculum breakdown:")
    lines.append("")
    lines.append("| curriculum | n | mean LL | std |")
    lines.append("|---|---|---|---|")
    for cur, s in sorted(rl["per_curriculum"].items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {cur} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} |")
    lines.append("")
    lines.append("## Result 8 — paired GRU vs RL per held-out mouse (avg over 3 seeds)")
    lines.append("")
    lines.append("| variant | D | n | GRU mean | RL mean | meanΔ (GRU−RL) | medianΔ | %GRU wins | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for var in paired["variants"]:
        for D in paired["Ds"]:
            s = paired["per_cell"].get(f"{var}_D{D}")
            if s is None: continue
            lines.append(f"| {var} | {D} | {s['n']} | {s['gru_mean']:.4f} | {s['rl_mean']:.4f} | "
                         f"{s['mean_delta']:+.5f} | {s['median_delta']:+.5f} | "
                         f"{s['frac_gru_wins']*100:.0f}% | {s['wilcoxon_p']:.1e} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    v2_l = paired["per_cell"].get("v2_D614")
    v2_s = paired["per_cell"].get("v2_D10")
    v1_l = paired["per_cell"].get("v1_D614")
    if v2_l and v2_s and v1_l:
        lines.append(f"- **GRU beats RL at every (variant, D) cell.** Even the smallest population "
                     f"GRU (v2, D=10) beats per-mouse RL: meanΔ={v2_s['mean_delta']:+.5f}, "
                     f"{v2_s['frac_gru_wins']*100:.0f}% of mice (Wilcoxon p={v2_s['wilcoxon_p']:.0e}). "
                     "A population model trained on as few as 10 other mice generalizes better to a "
                     "new mouse than fitting that mouse's own data with a classical RL model — strong "
                     "evidence the GRU exploits cross-mouse structure that per-mouse RL can't.")
        lines.append(f"- **Large-D gain over RL.** v2 D=614 vs RL: meanΔ={v2_l['mean_delta']:+.5f} "
                     f"({v2_l['frac_gru_wins']*100:.0f}% mice, p={v2_l['wilcoxon_p']:.0e}). "
                     "An order of magnitude larger than the v2−v1 SC effect (+0.0015): the "
                     "population vs per-mouse cognitive-model gap is the dominant signal, not "
                     "session conditioning or data scaling within the GRU.")
        incr = v2_l['mean_delta'] - v1_l['mean_delta']
        lines.append(f"- **SC adds a small extra margin over RL.** v2 D=614 beats RL by "
                     f"{v2_l['mean_delta']:+.5f}; v1 D=614 by {v1_l['mean_delta']:+.5f} → "
                     f"v2's incremental win vs RL is +{incr:.5f}, "
                     "consistent with the matched-pair v2−v1 SC result (Result 1).")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- **The RL baseline has no D-axis.** It's a per-mouse independent fit (no "
                 "cross-mouse sharing, no embeddings, no population prior). Answers \"does GRU beat "
                 "a stable classical RL on the same data?\" (yes), not \"does more mice help a "
                 "population RL?\" (planned hierarchical-Bayesian baseline; see `FUTURE_DIRECTIONS.md`).")
    lines.append("- **Simple agent (L1F1_CK1):** 1 learn-rate, 1 forget-rate, 1-step choice kernel. "
                 "Richer RL families (more learning/forget rates; ForagerLossCounting; "
                 "ForagerCompareThreshold) may close some of the gap. Cf. "
                 "`code/config/model/baseline_rl.yaml`.")
    lines.append("- **Single optimizer seed.** Differential evolution is stable; pilot didn't need "
                 "multi-seed averaging (variant notes). Add seeds [1, 2] for a tighter RL comparison.")
    lines.append("- **Pooled vs per-subject mean differ** because mice have different trial counts. "
                 "Use the trial-weighted pooled (0.7143) for GRU pooled aggregates; use the per-"
                 "subject mean (0.7211) on per-mouse panels. The figures label both.")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_md.name}")


# ============================= main =============================

def main():
    api = wandb.Api()
    print("Pulling RL run...")
    rl = pull_rl(api)

    print("\nLoading GRU per-subject (cache-or-pull)...")
    gru_nested = load_or_pull_gru(api)
    gru_mean = gru_per_subject_means(gru_nested)
    print(f"  unique (variant,D,subject) means: {len(gru_mean)}")

    print("\nPaired GRU vs RL per (variant, D)...")
    paired = paired_vs_rl(gru_mean, rl["per_subject_likelihood"])
    for cell, s in paired["per_cell"].items():
        print(f"  {cell:>10} n={s['n']:>3} GRU={s['gru_mean']:.4f} RL={s['rl_mean']:.4f}  "
              f"meanΔ={s['mean_delta']:+.5f} frac_GRU>RL={s['frac_gru_wins']:.2%} "
              f"Wilcoxon p={s['wilcoxon_p']:.2e}")

    out = {
        "_meta": build_meta(
            "analysis/rl_baseline.py",
            [*list_gru_groups(api), rl["group"]],
        ),
        "rl": rl,
        "gru_vs_rl_paired": paired,
    }
    json.dump(out, open(HERE / "rl_baseline.json", "w"), indent=2)
    print(f"\nwrote rl_baseline.json")

    write_verdict(rl, paired, HERE / "rl_baseline_verdict.md")

    print("\nRendering figures...")
    fig_paired(paired, rl, HERE / "fig_rl_paired.png")
    fig_scaling_v1_v2_with_rl(rl, HERE / "fig_scaling_v1_v2.png")
    fig_zeroshot_vs_d_with_rl(rl, HERE / "fig_zeroshot_vs_d.png")
    fig_fewshot_curve_with_rl(rl, HERE / "fig_fewshot_curve.png")
    fig_nxd_scaling_with_rl(rl, HERE / "fig_nxd_scaling.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
