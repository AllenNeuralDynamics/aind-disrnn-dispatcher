#!/usr/bin/env python
"""Update Result 7 in FINAL_REPORT.md from nxd_scaling.json.

This intentionally rewrites only the N x D section, bounded by the Result 7
heading and the following "On effect sizes" heading.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).parent
REPORT = HERE / "FINAL_REPORT.md"
NXD_JSON = HERE / "nxd_scaling.json"
START = "## Result 7 — N × D joint scaling grid (Chinchilla-style)"
END = "## Result 8"  # next section header; bounds the Result 7 block tightly


def _fmt(v: float) -> str:
    if v is None or math.isnan(v):
        return "nan"
    return f"{v:.4f}"


def _d_index(ds: list[int], d: int) -> int:
    return ds.index(d)


def _cell(grid: list[list[float]], ns: list[int], ds: list[int], n: int, d: int) -> float:
    return grid[ns.index(n)][_d_index(ds, d)]


def _pct(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v * 100:.0f}%"


def build_block(data: dict) -> str:
    ns = data["Ns"]
    ds = data["Ds"]
    grid = data["mean_grid"]
    fit_add = data["fit_additive"]
    fit_int = data["fit_interaction"]
    fit_ll = data.get("fit_loglog_interaction", {})
    groups = ", ".join(f"`{g}`" for g in data["groups"])

    d_cols = " | ".join(f"D={d}" for d in ds)
    align = "|---|" + "---|" * (len(ds) + 2)
    rows = []
    sat_fracs = []
    for i, n in enumerate(ns):
        vals = [grid[i][j] for j in range(len(ds))]
        l10 = _cell(grid, ns, ds, n, 10)
        l100 = _cell(grid, ns, ds, n, 100)
        l614 = _cell(grid, ns, ds, n, 614)
        tot = l614 - l10
        late = l614 - l100
        frac = (l100 - l10) / tot if abs(tot) > 1e-9 else float("nan")
        sat_fracs.append(frac)
        row_vals = " | ".join(_fmt(v) for v in vals)
        rows.append(f"| {n} | {row_vals} | {late:+.4f} | {_pct(frac)} |")

    mean_frac = sum(sat_fracs) / len(sat_fracs)
    diff_d10 = _cell(grid, ns, ds, ns[-1], 10) - _cell(grid, ns, ds, ns[0], 10)
    diff_d614 = _cell(grid, ns, ds, ns[-1], 614) - _cell(grid, ns, ds, ns[0], 614)
    ratio = diff_d614 / diff_d10 if abs(diff_d10) > 1e-9 else float("nan")

    d_aic = fit_int["aic"] - fit_add["aic"] if "error" not in fit_int and "error" not in fit_add else float("nan")
    interaction = fit_ll.get("interaction", {})
    p = interaction.get("p")
    p_text = "nan" if p is None else f"{p:.3f}"

    block = [
        START,
        "![N x D joint scaling](fig_nxd_scaling.png)",
        "",
        "*Heatmap and paired slices through the N×D grid. D saturates by ~100 mice at each hidden size, while the fixed-D N gain grows modestly from D=10 to D=614.*",
        "",
        f"Grid: N (hidden_size) ∈ {{{', '.join(str(n) for n in ns)}}} × D ∈ {{{', '.join(str(d) for d in ds)}}} × 3 seeds ({len(ns) * len(ds)} (N,D) cells). H128 column re-used from `v2-sc-active`; D=30 for H16/H64/H256 comes from the g6e gap-fill. Metric: aggregate `heldout/final/eval_likelihood` across the same fixed held-out mouse set (~149 mice).",
        "",
        "Mean L grid (held-out eval likelihood):",
        "",
        f"| N | {d_cols} | Δ (D100→D614) | frac of D-gain by D=100 |",
        align,
        *rows,
        "",
        f"- *D saturates by ~100 across every N tested* (mean {mean_frac * 100:.0f}% of D-gain captured by D=100). Saturation is *not* a hidden-size artifact — it persists from H=16 to H=256.",
        f"- *N-axis gain at fixed D grows weakly with D* (Chinchilla-style interaction). N={ns[0]}→{ns[-1]} gain: {diff_d10:+.4f} at D=10, {diff_d614:+.4f} at D=614 ({ratio:.1f}×). Qualitative support for an N×D synergy, but absolute magnitudes are small (<0.01 nats/trial).",
        f"- Additive fit `L = E + A·N^{{-α}} + B·D^{{-β}}`: E≈{fit_add['E']:.3f} (single irreducible floor), α≈{fit_add['alpha']:.2f} (N), β≈{fit_add['beta']:.2f} (D); {'D-axis dominates' if abs(fit_add['beta']) > abs(fit_add['alpha']) else 'N-axis dominates'} within this grid.",
        f"- Interaction-term fit ΔAIC vs additive: {d_aic:+.1f}; log-log interaction p={p_text}. Treat the nonlinear interaction fit as descriptive because the grid remains small relative to the number of fit parameters.",
        f"- *Verdict*: same predictability ceiling story as Result 1; adding D=30 fills the low-data bend but does not by itself create new headroom. RL reference (trial-weighted pooled **0.7143**, dashed line on slice panels) sits below every (N, D) cell. See `nxd_scaling_verdict.md` for the fit details.",
        "",
        f"Source W&B groups: {groups}.",
        "",
    ]
    return "\n".join(block)


def update_report(dry_run: bool = False) -> str:
    data = json.loads(NXD_JSON.read_text())
    block = build_block(data)
    text = REPORT.read_text()
    start = text.index(START)
    end = text.index(END, start)
    updated = text[:start] + block + text[end:]
    if not dry_run:
        REPORT.write_text(updated)
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    block = update_report(dry_run=args.dry_run)
    if args.dry_run:
        print(block)
    else:
        print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
