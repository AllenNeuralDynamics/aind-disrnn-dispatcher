#!/usr/bin/env python
"""Joint N×D scaling: held-out-mouse LL vs (hidden_size N × #training-mice D).

Grid (SC-active, session_encoding=scalar, multisubject): N∈{16,64,256} × D∈{10,100,614},
3 seeds (nxd-grid@... group). The N=128 row is the main-study v2-SC-active H128 D-sweep
(paired_v1_v2_cell.json), same recipe/heldout cohort — grafted in for a 4×3 grid.

Tests the Chinchilla question: is there an N×D INTERACTION (does capacity only pay off with
more data, and vice versa)? Fits (a) a log-log regression with interaction term and (b) the
separable saturating form L = E - A·N^-alpha - B·D^-beta. Run with the wrapper venv.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import wandb
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PROJECT = "AIND-disRNN/mice_data_scaling"
DLAB = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}


def _g(cfg, *path):
    d = cfg
    for p in path:
        d = (d or {}).get(p) if isinstance(d, dict) else None
    return d


def collect_grid():
    api = wandb.Api()
    runs = [r for r in api.runs(PROJECT)
            if (r.group or "").startswith("nxd-grid@") and r.state == "finished"]
    by = {}
    for r in runs:
        N = _g(r.config, "model", "architecture", "hidden_size")
        ratio = _g(r.config, "data", "subject_ratio")
        seed = _g(r.config, "data", "subject_sample_seed")
        key = (int(N), round(float(ratio), 3), seed)
        if key not in by or str(r.created_at) > str(by[key].created_at):
            by[key] = r
    rows = []  # (N, D, seed, ll)
    for (N, ratio, seed), r in by.items():
        ll = r.summary.get("heldout/final/eval_likelihood") or r.summary.get("heldout/eval_likelihood")
        rows.append((N, DLAB.get(ratio, ratio), seed, float(ll)))
    return rows


def main():
    rows = collect_grid()
    print(f"grid: {len(rows)} runs")
    # add H128 (v2 SC-active main study), per-D cell means
    paired = json.load(open(HERE / "paired_v1_v2_cell.json"))
    pr = paired.get("per_ratio", paired)
    for k, v in pr.items():
        D = DLAB.get(round(float(k), 3))
        if D in (10, 100, 614) and isinstance(v, dict) and "v2" in v:
            rows.append((128, D, "h128main", float(v["v2"])))

    # cell means
    cell = defaultdict(list)
    for N, D, seed, ll in rows:
        cell[(N, D)].append(ll)
    Ns = sorted({n for n, _ in cell})
    Ds = sorted({d for _, d in cell})
    L = {k: float(np.mean(v)) for k, v in cell.items()}

    # --- (a) log-log regression with interaction ---
    # LL ~ b0 + b1*lnN + b2*lnD + b3*lnN*lnD  (use cell means, centered logs)
    lnN0 = np.log(np.array(Ns)).mean(); lnD0 = np.log(np.array(Ds)).mean()
    X, y = [], []
    for (N, D), m in L.items():
        a, b = np.log(N) - lnN0, np.log(D) - lnD0
        X.append([1, a, b, a * b]); y.append(m)
    X = np.array(X); y = np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    names = ["intercept", "lnN", "lnD", "lnN*lnD (interaction)"]
    reg = {n: dict(coef=float(b), se=float(s), t=float(t), p=float(p))
           for n, b, s, t, p in zip(names, beta, se, tvals, pvals)}

    # --- (b) separable saturating fit L = E - A*N^-alpha - B*D^-beta ---
    def model(P, N, D):
        E, A, al, B, be = P
        return E - A * N ** (-al) - B * D ** (-be)
    NN = np.array([k[0] for k in L]); DD = np.array([k[1] for k in L]); yy = np.array(list(L.values()))
    try:
        popt, _ = optimize.curve_fit(
            lambda X, E, A, al, B, be: model((E, A, al, B, be), X[0], X[1]),
            np.vstack([NN, DD]), yy,
            p0=[0.74, 0.1, 0.3, 0.1, 0.3], maxfev=20000)
        E, A, al, B, be = [float(x) for x in popt]
        pred = model(popt, NN, DD)
        r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        sep = dict(E=E, A=A, alpha=al, B=B, beta=be, r2=float(r2))
    except Exception as e:
        sep = {"error": str(e)}

    # empirical interaction: ΔD across N, ΔN across D
    dD = {N: L[(N, max(Ds))] - L[(N, min(Ds))] for N in Ns if (N, max(Ds)) in L and (N, min(Ds)) in L}
    dN = {D: L[(max(Ns), D)] - L[(min(Ns), D)] for D in Ds if (max(Ns), D) in L and (min(Ns), D) in L}

    out = dict(
        grid={f"N{N}_D{D}": dict(ll=L[(N, D)], n=len(cell[(N, D)])) for (N, D) in sorted(L)},
        Ns=Ns, Ds=Ds, regression=reg, separable_fit=sep,
        deltaD_by_N=dD, deltaN_by_D=dN)
    json.dump(out, open(HERE / "nxd_result.json", "w"), indent=2)

    # figure: LL vs D, one line per N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    cmap = plt.get_cmap("viridis")
    for i, N in enumerate(Ns):
        xs = [D for D in Ds if (N, D) in L]
        ax1.plot(xs, [L[(N, D)] for D in xs], "o-", color=cmap(i / max(1, len(Ns) - 1)), label=f"N={N}")
    ax1.set_xscale("log"); ax1.set_xlabel("# training mice (D)"); ax1.set_ylabel("held-out LL")
    ax1.set_title("LL vs D, by capacity N"); ax1.legend()
    # heatmap of LL
    M = np.array([[L.get((N, D), np.nan) for D in Ds] for N in Ns])
    im = ax2.imshow(M, aspect="auto", origin="lower", cmap="viridis")
    ax2.set_xticks(range(len(Ds))); ax2.set_xticklabels(Ds); ax2.set_xlabel("D")
    ax2.set_yticks(range(len(Ns))); ax2.set_yticklabels(Ns); ax2.set_ylabel("N (hidden_size)")
    ax2.set_title("held-out LL(N, D)"); fig.colorbar(im, ax=ax2)
    fig.tight_layout(); fig.savefig(HERE / "fig_nxd_result.png", dpi=150); plt.close(fig)

    print("\n=== L(N,D) ===")
    print("  N\\D " + "  ".join(f"{d:>7}" for d in Ds))
    for N in Ns:
        print(f"  {N:>4} " + "  ".join(f"{L.get((N,D),float('nan')):.4f}" for D in Ds))
    print("\nΔD(min→max D) by N:", {k: round(v, 4) for k, v in dD.items()})
    print("ΔN(min→max N) by D:", {k: round(v, 4) for k, v in dN.items()})
    print("\ninteraction coef lnN*lnD: "
          f"{reg['lnN*lnD (interaction)']['coef']:+.5f} (p={reg['lnN*lnD (interaction)']['p']:.3f})")
    print("separable fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sep.items()})


if __name__ == "__main__":
    main()
