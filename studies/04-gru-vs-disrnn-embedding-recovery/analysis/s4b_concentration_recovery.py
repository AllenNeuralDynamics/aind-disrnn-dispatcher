#!/usr/bin/env python
"""Stage-4b concentration sweep — recovery vs Dirichlet mixture concentration.

Facets the in-container recovery output (`s4b_concentration_recovery.json`, produced
by aind-disrnn-wrapper `code/analysis/stage4b_recovery.py` on the
`gru-stage4b-concentration@20260712-060852` runs) by the swept mixture concentration,
and adds a mechanism diagnostic: how much *session position* the session-conditioned
embedding encodes vs. the per-session *family* it fails to.

Question this closes: `gru-stage4b` found "session conditioning adds nothing." Is that
a Dirichlet(0.5)-too-sparse artifact, or fundamental? Sweeping concentration
{0.5,1,2,5} (sparse → near-uniform mixtures): the session-conditioned vs subject-only
per-session decoding gap stays ~0 at every concentration → **fundamental**. The
mechanism: the session embedding encodes smooth session *position* (R²≈0.6) but not the
discrete per-session family (gap≈0) — a continuous position code cannot represent a
discrete i.i.d. regime draw.

Inputs
------
- `analysis/s4b_concentration_recovery.json`  (committed; in-container recovery output)
- `analysis/s4b_conc_inventory.json`          (committed; run -> data_cfg incl. concentration)
- session-conditioned embedding CSVs `ctx_<run>.csv` for the position-R² panel. These
  are large per-run W&B-pull artifacts (gitignored) — repopulate from the recovery
  job's Beaker result dataset:
      beaker dataset fetch 01KXBYQV5P9NSPWQ72WG7G2A87 -o <CTX_DIR>/../
  Pass --ctx-dir <dir containing ctx_*.csv>. If absent, the position-R² panel/column is
  skipped and the rest still regenerates from the committed grid.

Outputs
-------
- `analysis/s4b_concentration_grid.csv`   (curated per-(conc,enc) summary)
- `analysis/figures/fig_s4b_concentration.png`
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "util"))
from plot_style import apply_presentation_style, t975  # noqa: E402

CONCS = [0.5, 1.0, 2.0, 5.0]
N_FAMILIES = 3
CHANCE = 1.0 / N_FAMILIES


def _conc(inv_entry: dict) -> float:
    d = inv_entry["data_cfg"]
    return ((d["agent"].get("subject_presets") or {}).get("session_switching") or {}).get("concentration")


def _position_r2(ctx_dir: Path, inv: dict) -> dict[float, list[float]]:
    """R² of predicting session position (session_phase) from the session-conditioned
    embedding, cross-validated by subject. High => the session code is a position code."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import r2_score

    out: dict[float, list[float]] = collections.defaultdict(list)
    for ctx in sorted(ctx_dir.glob("ctx_*.csv")):
        rid = ctx.name[4:-4]
        if rid not in inv:
            continue
        df = pd.read_csv(ctx)
        ec = [c for c in df.columns if c.startswith("embedding_")]
        yhat = cross_val_predict(
            LinearRegression(), df[ec].values, df["session_phase"].values,
            groups=df["subject_index"].values, cv=GroupKFold(5),
        )
        out[_conc(inv[rid])].append(r2_score(df["session_phase"].values, yhat))
    return out


def build_grid(rec: dict, inv: dict, pos_r2: dict[float, list[float]]) -> pd.DataFrame:
    rows = []
    for rid, m in rec.items():
        c = _conc(inv[rid])
        rows.append({
            "concentration": c, "enc": m["enc"], "seed": inv[rid]["data_cfg"]["seed"],
            "mix_r2_mean": m.get("mix_r2_mean"),
            "dominant_family_acc": m.get("dominant_family_acc"),
            "persession_sessioncond": m.get("persession_family_acc_sessioncond"),
            "persession_subjectonly": m.get("persession_family_acc_subjectonly"),
        })
    per_run = pd.DataFrame(rows)

    def agg(sub, col):
        v = sub[col].dropna().values
        n = len(v)
        return (float(np.mean(v)) if n else np.nan,
                float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0, n)

    out = []
    for c in CONCS:
        for enc in ["none", "scalar"]:
            sub = per_run[(per_run.concentration == c) & (per_run.enc == enc)]
            row = {"concentration": c, "enc": enc}
            for col in ["mix_r2_mean", "dominant_family_acc",
                        "persession_sessioncond", "persession_subjectonly"]:
                mean, sem, n = agg(sub, col)
                row[col] = mean
                row[f"{col}_sem"] = sem
                row["n_seeds"] = n
            if enc == "scalar" and c in pos_r2:
                pr = pos_r2[c]
                row["position_r2"] = float(np.mean(pr))
                row["position_r2_sem"] = float(np.std(pr, ddof=1) / np.sqrt(len(pr))) if len(pr) > 1 else 0.0
            out.append(row)
    return pd.DataFrame(out)


def make_figure(grid: pd.DataFrame, out_png: Path, has_pos: bool) -> None:
    import matplotlib.pyplot as plt
    apply_presentation_style()
    sc = grid[grid.enc == "scalar"].sort_values("concentration")
    no = grid[grid.enc == "none"].sort_values("concentration")
    x = sc.concentration.values
    xi = np.arange(len(x))                       # categorical positions (even spacing)
    labels = [f"{c:g}" for c in x]

    def _catx(ax):
        ax.set_xticks(xi); ax.set_xticklabels(labels)
        ax.set_xlim(-0.3, len(x) - 0.7)
        ax.set_xlabel("Dirichlet concentration α")

    n_panels = 3 if has_pos else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5.0))

    # Panel A — per-session family decoding: session-conditioned vs subject-only ~ overlap
    ax = axes[0]
    ax.errorbar(xi, sc.persession_sessioncond, yerr=sc.persession_sessioncond_sem,
                marker="o", label="session-conditioned emb", color="#C44E52", capsize=3)
    ax.errorbar(xi, sc.persession_subjectonly, yerr=sc.persession_subjectonly_sem,
                marker="s", label="subject-only emb", color="#4C72B0", capsize=3)
    ax.axhline(CHANCE, ls=":", color="0.5", label="chance (1/3)")
    _catx(ax); ax.set_ylabel("per-session family decoding acc")
    ax.set_title("Session conditioning adds nothing\n(gap ≈ 0 at every α)")
    ax.legend(frameon=False, fontsize=12)

    # Panel B — subject-level mixture recovery falls with mixing
    ax = axes[1]
    ax.errorbar(xi, sc.mix_r2_mean, yerr=sc.mix_r2_mean_sem, marker="o",
                color="#55A868", capsize=3, label="scalar")
    ax.errorbar(xi, no.mix_r2_mean, yerr=no.mix_r2_mean_sem, marker="s",
                color="#55A868", alpha=0.45, capsize=3, label="none")
    ax.axhline(0, ls="-", color="0.7", lw=1)
    _catx(ax); ax.set_ylabel("subject mix-weight R²")
    ax.set_title("Subject identity gets\nun-identifiable as α → uniform")
    ax.legend(frameon=False, fontsize=12)

    # Panel C — the mechanism: session code encodes POSITION, not family
    if has_pos:
        ax = axes[2]
        gap = (sc.persession_sessioncond.values - sc.persession_subjectonly.values)
        ax.errorbar(xi, sc.position_r2, yerr=sc.position_r2_sem, marker="o",
                    color="#8172B3", capsize=3, label="R²: emb → session position")
        ax.plot(xi, gap, marker="D", color="#C44E52", label="family gap (SC − subj)")
        ax.axhline(0, ls="-", color="0.7", lw=1)
        _catx(ax); ax.set_ylim(-0.15, 0.85); ax.set_ylabel("R² / accuracy gap")
        ax.set_title("The session code encodes\nPOSITION, not family")
        ax.legend(frameon=False, fontsize=12)

    fig.suptitle("Stage-4b · GRU · per-session family switching vs mixture concentration "
                 "(N=200, 2 seeds; error bars SEM)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recovery", default=str(HERE / "s4b_concentration_recovery.json"))
    p.add_argument("--inventory", default=str(HERE / "s4b_conc_inventory.json"))
    p.add_argument("--ctx-dir", default=None,
                   help="dir with ctx_<run>.csv for the position-R² panel (see module docstring)")
    p.add_argument("--out-grid", default=str(HERE / "s4b_concentration_grid.csv"))
    p.add_argument("--out-fig", default=str(HERE / "figures" / "fig_s4b_concentration.png"))
    args = p.parse_args()

    rec = json.load(open(args.recovery))
    inv = json.load(open(args.inventory))
    pos_r2: dict[float, list[float]] = {}
    if args.ctx_dir and Path(args.ctx_dir).is_dir():
        pos_r2 = _position_r2(Path(args.ctx_dir), inv)
        print(f"position-R² from {sum(len(v) for v in pos_r2.values())} scalar ctx CSVs")
    else:
        print("no --ctx-dir; skipping position-R² panel (mechanism)")

    grid = build_grid(rec, inv, pos_r2)
    grid.to_csv(args.out_grid, index=False)
    print(f"wrote {args.out_grid}")
    make_figure(grid, Path(args.out_fig), has_pos=bool(pos_r2))

    # console summary
    print("\nper-session family decoding — session-conditioned vs subject-only (mean/2 seeds):")
    sc = grid[grid.enc == "scalar"].sort_values("concentration")
    for _, r in sc.iterrows():
        gap = r.persession_sessioncond - r.persession_subjectonly
        pr = f" | posR²={r.position_r2:.3f}" if "position_r2" in r and pd.notna(r.get("position_r2")) else ""
        print(f"  α={r.concentration:>3}: SC={r.persession_sessioncond:.3f} "
              f"subj={r.persession_subjectonly:.3f} gap={gap:+.3f}{pr}")


if __name__ == "__main__":
    main()
