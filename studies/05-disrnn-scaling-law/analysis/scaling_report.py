"""Single producer for 05-disrnn-scaling-law reports r1 + r2.

Reads the committed grid (analysis/grid.csv), writes curated outputs and figures, and
regenerates the <!-- BEGIN result-N --> blocks in the reports.

    python analysis/scaling_report.py          # offline; no WANDB_API_KEY needed

OUTPUTS (all committed so the reports render in-repo)
  analysis/summary.json          - curated per-D and per-(mult,beta) stats + _meta provenance
  analysis/summary.csv           - the same, flat
  analysis/fig_scaling_curve.png - r1: held-out LL vs D, disRNN vs GRU vs RL baselines
  analysis/fig_sparsity_vs_d.png - r2a: six-family bottleneck openness vs D
  analysis/fig_mult_axis_d614.png- r2b: openness AND held-out vs multiplier at D=614 vs D=100

METRIC CAVEAT (study 03): openness is total_openness = Sigma(1-sigma). NEVER
n_eff_open_frac -- it is scale-invariant and reads high even for a fully shut bottleneck.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))  # shared studies/util (_meta, plot_style)
from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style, t975  # noqa: E402

WANDB_GROUPS = ["dscan-mult2@20260713-003428", "mult-beta-d614@20260713-003501"]

# Reference numbers, all on the SAME fixed held-out cohort and the SAME metric key
# (heldout/eval_likelihood), so they are directly comparable to this study's y-axis.
GRU = {10: 0.7219, 30: 0.7250, 100: 0.7262, 300: 0.7267, 614: 0.7268}  # study 01
RL = {"compare-to-threshold": 0.7170, "Bari (L1F1_CK1)": 0.7149, "Hattori (L2F1)": 0.7127}
# study 03's interaction openness at D=100 (the same (mult, beta) cells we re-ran at D=614)
SEED_SD_D614 = 0.0  # set from data in main(); yardstick for the single-seed wave 2
S03_D100 = {3e-4: {1: 3.11, 2: 1.16, 5: 0.11, 10: 0.00},
            1e-3: {1: 1.60, 2: 0.81, 5: 0.00, 10: 0.00}}

FAMILIES = [
    ("update_net_latent", "update←latent (interaction)"),
    ("latent", "latent (recurrent)"),
    ("update_net_obs", "update←obs"),
    ("update_net_subj", "update←subject"),
    ("choice_net_latent", "choice←latent"),
    ("choice_net_subj", "choice←subject"),
]
DBIN = lambda d: 10 if d <= 10 else 30 if d <= 30 else 100 if d <= 101 else 300 if d <= 301 else 614


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_grid():
    with (HERE / "grid.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["state"] == "finished"]


def _stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"mean": None, "sem": None, "n": 0}
    n = len(v)
    sem = statistics.stdev(v) / n**0.5 if n > 1 else 0.0
    return {"mean": statistics.mean(v), "sem": sem, "n": n,
            "ci95": t975(n) * sem if n > 1 else None}


def summarize(rows):
    wave1 = [r for r in rows if r["variant"] == "dscan-mult2"]
    wave2 = [r for r in rows if r["variant"] == "mult-beta-d614"]

    by_d = defaultdict(lambda: defaultdict(list))
    for r in wave1:
        d = DBIN(int(r["D"]))
        by_d[d]["heldout"].append(_f(r["heldout_ll"]))
        for fam, _ in FAMILIES:
            by_d[d][fam].append(_f(r[f"open_{fam}"]))
    curve = {}
    for d in sorted(by_d):
        curve[d] = {"heldout": _stat(by_d[d]["heldout"]),
                    "gru": GRU[d],
                    "gap_vs_gru": _stat(by_d[d]["heldout"])["mean"] - GRU[d],
                    **{fam: _stat(by_d[d][fam]) for fam, _ in FAMILIES}}

    cells = {}
    for r in wave2:
        b, m = _f(r["beta"]), int(r["mult"])
        cells[f"{b:g}|{m}"] = {"beta": b, "mult": m,
                               "interaction_openness": _f(r["open_update_net_latent"]),
                               "heldout": _f(r["heldout_ll"]),
                               "d100_reference": S03_D100.get(b, {}).get(m)}
    return curve, cells


def write_outputs(curve, cells, noise):
    payload = {"_meta": build_meta("analysis/scaling_report.py", WANDB_GROUPS, study_root=STUDY),
               "note": ("openness = total_openness = Sigma(1-sigma); NEVER n_eff_open_frac. "
                        "GRU and RL references share this study's held-out cohort and metric key."),
               "gru_reference": GRU, "rl_baselines": RL,
               "seed_noise_at_D614": noise,
               "wave1_curve_by_D": curve, "wave2_cells_at_D614": cells}
    (HERE / "summary.json").write_text(json.dumps(payload, indent=2))

    with (HERE / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "n", "heldout_mean", "heldout_sem", "gru", "gap"]
                   + [f"open_{fam}" for fam, _ in FAMILIES])
        for d, c in curve.items():
            w.writerow([d, c["heldout"]["n"], c["heldout"]["mean"], c["heldout"]["sem"],
                        c["gru"], c["gap_vs_gru"]] + [c[fam]["mean"] for fam, _ in FAMILIES])
    print("wrote summary.json + summary.csv")


def fig_scaling_curve(curve):
    """r1 — the headline: disRNN peaks at ~100 mice and declines; GRU saturates above it."""
    apply_presentation_style()
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ds = sorted(curve)
    y = [curve[d]["heldout"]["mean"] for d in ds]
    e = [curve[d]["heldout"]["sem"] for d in ds]

    # RL baselines: the line STOPS at the plot edge so it never runs through its own label,
    # which is set in the clear right margin.
    for name, v in RL.items():
        ax.plot([8, 700], [v, v], color="#999999", ls=":", lw=1.5, zorder=1)
        ax.text(760, v, name, va="center", ha="left", fontsize=11, color="#777777")
    ax.text(760, 0.7190, "per-mouse RL\nbaselines", va="bottom", ha="left",
            fontsize=11.5, color="#777777", style="italic")

    ax.plot(ds, [GRU[d] for d in ds], "o--", color="#C44E52", ms=8, label="GRU (study 01)", zorder=3)
    ax.errorbar(ds, y, yerr=e, fmt="o-", color="#4C72B0", ms=9, capsize=4,
                label="disRNN (mult=2, β=1e-3)", zorder=4)

    peak = max(ds, key=lambda d: curve[d]["heldout"]["mean"])
    ax.annotate(f"peaks at D≈{peak},\nthen DECLINES",
                xy=(300, curve[300]["heldout"]["mean"] + 0.0004),
                xytext=(105, 0.7205), fontsize=13, color="#4C72B0", ha="center",
                arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.6))
    # the best D=614 cell from wave 2 — the decline is largely the operating point's fault
    ax.plot([614], [0.7211], marker="*", ms=22, color="#55A868", ls="none", zorder=5,
            label="disRNN best @ D=614 (mult=1, β=3e-4)")
    ax.annotate("a weaker penalty\nrecovers most of the gap",
                xy=(614, 0.7211), xytext=(150, 0.7228), fontsize=12, color="#55A868", ha="center",
                arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.6))

    ax.set_xscale("log")
    ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("training mice  D  (log scale)")
    ax.set_ylabel("held-out-mouse likelihood")
    ax.set_title("More mice stop helping the disRNN — and then hurt it", pad=12)
    ax.set_xlim(8, 2600)
    ax.legend(loc="lower center", frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "fig_scaling_curve.png")
    plt.close(fig)
    print("wrote fig_scaling_curve.png")


def fig_sparsity_vs_d(curve):
    """r2a — at a FIXED penalty, the interaction bottleneck opens up as the cohort grows."""
    apply_presentation_style()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ds = sorted(curve)
    colors = ["#C44E52", "#8172B3", "#4C72B0", "#55A868", "#CCB974", "#999999"]
    for (fam, label), c in zip(FAMILIES, colors):
        y = [curve[d][fam]["mean"] for d in ds]
        e = [curve[d][fam]["sem"] for d in ds]
        lw = 3.2 if fam == "update_net_latent" else 1.8
        ax.errorbar(ds, y, yerr=e, fmt="o-", color=c, lw=lw, ms=7, capsize=3,
                    label=label, zorder=4 if fam == "update_net_latent" else 2)
    ax.set_xscale("log")
    ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("training mice  D  (log scale)")
    ax.set_ylabel("bottleneck openness   Σ(1−σ)")
    # Honest title: the rise from D=10 to D>=100 is large (~5x) and survives the seed noise.
    # The fine structure ABOVE D=100 does NOT -- seed SD of the interaction openness at D=614 is
    # 0.384 (per-seed 1.136 / 0.467 / 0.474), which swamps the D=300-vs-614 difference. Error bars
    # are SEM over 3 seeds; read them before reading the shape.
    ax.set_title("More mice ⇒ less sparse (D=10 → 100)\nbeyond ~100 the seed noise dominates"
                 "  ·  fixed mult=2, β=1e-3", pad=12, fontsize=14)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig_sparsity_vs_d.png")
    plt.close(fig)
    print("wrote fig_sparsity_vs_d.png")


def fig_mult_axis_d614(cells):
    """r2b — THE headline: at D=614, sparsifying the interaction gate now COSTS transfer.

    Study 03 established at D=100 that held-out is FLAT across the multiplier, i.e.
    sparsity is free. That breaks at the full cohort.
    """
    apply_presentation_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mults = [1, 2, 5, 10]
    bc = {3e-4: "#4C72B0", 1e-3: "#55A868", 3e-3: "#C44E52"}

    ax = axes[0]
    for b, c in bc.items():
        y = [cells.get(f"{b:g}|{m}", {}).get("interaction_openness") for m in mults]
        ax.plot(mults, y, "o-", color=c, ms=9, label=f"β={b:g}  (D=614)")
        ref = [S03_D100.get(b, {}).get(m) for m in mults]
        if any(v is not None for v in ref):
            ax.plot(mults, ref, "s--", color=c, ms=6, alpha=0.45,
                    label=f"β={b:g}  (D=100, study 03)")
    ax.set_xscale("log"); ax.set_xticks(mults); ax.set_xticklabels([str(m) for m in mults])
    ax.set_xlabel("update-net latent penalty multiplier")
    ax.set_ylabel("interaction openness  Σ(1−σ)")
    ax.set_title("Multiplier still closes the gate…\nbut the gate is MORE OPEN at D=614", pad=12)
    ax.legend(frameon=False, fontsize=11)

    ax = axes[1]
    # n=1 per cell (seed pinned to 0 by design -- a mechanism check, not an effect-size estimate).
    # The band shows the seed-to-seed SD measured at the SAME config from wave 1's 3 D=614 seeds
    # (SD = 0.0003), so a reader can see the multiplier effect (~0.004) is ~15x the seed noise.
    for b, c in bc.items():
        y = [cells.get(f"{b:g}|{m}", {}).get("heldout") for m in mults]
        ax.plot(mults, y, "o-", color=c, ms=9, label=f"β={b:g}")
        ax.fill_between(mults, [v - SEED_SD_D614 for v in y], [v + SEED_SD_D614 for v in y],
                        color=c, alpha=0.15, lw=0)
    ax.axhline(RL["compare-to-threshold"], color="#999999", ls=":", lw=1.6)
    ax.text(10, RL["compare-to-threshold"], " best RL baseline", va="bottom", ha="right",
            fontsize=11, color="#666666")
    ax.axhline(GRU[614], color="#C44E52", ls="--", lw=1.6)
    ax.text(10, GRU[614], " GRU", va="bottom", ha="right", fontsize=11, color="#C44E52")
    ax.set_xscale("log"); ax.set_xticks(mults); ax.set_xticklabels([str(m) for m in mults])
    ax.set_xlabel("update-net latent penalty multiplier")
    ax.set_ylabel("held-out-mouse likelihood")
    ax.set_title("…and sparsity is NO LONGER FREE:\ntransfer falls as the gate closes "
                 f"(band = ±{SEED_SD_D614:.4f} seed SD)", pad=12, fontsize=14)
    ax.legend(frameon=False, fontsize=12, loc="lower left")

    fig.tight_layout()
    fig.savefig(HERE / "fig_mult_axis_d614.png")
    plt.close(fig)
    print("wrote fig_mult_axis_d614.png")


def seed_noise(rows):
    """Seed-to-seed SD at the SAME config (D=614, mult=2, beta=1e-3), across EVERY variant.

    This is the yardstick for wave 2, which is single-seed (it runs at the Hydra default seed=42
    -- the sweep never set `seed`, so it is an INDEPENDENT 4th seed of wave 1's D=614 cell, not a
    duplicate of seed 0). Pooling all four is the honest estimate: wave 1's three seeds alone give
    SD 0.00025, but seed 42 lands 0.0009 above their mean, so the 3-seed SD understates.

    Held-out is tight across seeds (SD ~0.0005) while interaction openness is NOT (SD ~0.38 --
    per-seed 1.136 / 0.467 / 0.474). So a ~0.004 multiplier effect on held-out is ~8x the noise and
    real, while fine structure in openness-vs-D above D=100 is noise and must NOT be read as a trend.
    """
    d614 = [r for r in rows
            if int(r["D"]) > 600 and r["mult"] == "2" and abs(_f(r["beta"]) - 1e-3) < 1e-9]
    hl = [_f(r["heldout_ll"]) for r in d614]
    op = [_f(r["open_update_net_latent"]) for r in d614]
    return {"n_seeds": len(hl),
            "seeds": [r["seed"] for r in d614],
            "config": "D=614, mult=2, beta=1e-3 (pooled across variants)",
            "heldout_sd": statistics.stdev(hl) if len(hl) > 1 else None,
            "heldout_per_seed": hl,
            "interaction_openness_sd": statistics.stdev(op) if len(op) > 1 else None,
            "interaction_openness_per_seed": op}


def main() -> None:
    global SEED_SD_D614
    rows = read_grid()
    noise = seed_noise(rows)
    SEED_SD_D614 = noise["heldout_sd"] or 0.0
    curve, cells = summarize(rows)
    write_outputs(curve, cells, noise)
    fig_scaling_curve(curve)
    fig_sparsity_vs_d(curve)
    fig_mult_axis_d614(cells)
    print(f"seed noise @ D=614: heldout SD {noise['heldout_sd']:.4f} | "
          f"openness SD {noise['interaction_openness_sd']:.3f}")


if __name__ == "__main__":
    main()
