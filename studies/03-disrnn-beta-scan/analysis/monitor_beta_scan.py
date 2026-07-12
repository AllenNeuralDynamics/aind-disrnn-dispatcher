#!/usr/bin/env python
"""Live monitor for the disRNN update-net-ratio beta-scan grid.

Pulls the current loss + bottleneck-sparsity state for every run in the W&B
project, grouped by the grid axes (multiplier x base-beta x lr x seed), and
optionally renders trajectory plots for the two metrics we steer on:
  - train/loss                                  (fit)
  - bottlenecks/update_net_latent_frac_open     (the multiplier's direct target)
  - bottlenecks/total_sigma                     (overall bottleneck tightness)

WHY: the disRNN trainer has NO early stopping (unlike the GRU trainer, which
stops on a loss metric). For this study we deliberately train a fixed 150k main
steps so the sparsity-vs-multiplier comparison is at a common horizon. This
script is the manual substitute: run it any time to see how far each run has
progressed and whether loss AND sparsity have BOTH stabilized -- the joint
"everything settled" condition that a future sparsity-aware early stop would key
on. Do NOT read a loss plateau alone as "done": the bottleneck keeps closing
(frac_open 1->0) well after loss flattens, and that closing IS the result.

The `_step` axis includes the n_warmup_steps offset (default 7500): _step < 7500
is warmup (bottlenecks held open); main training starts at _step = 7500 and runs
to 7500 + n_steps.

Sandbox-safe: uses the W&B GraphQL endpoint with WANDB_API_KEY (no wandb SDK
login). The multiplier is consumed pre-training by resolve_disrnn_penalties and
NOT logged directly, so it is recovered as round(update_net_latent_penalty/beta).

Usage:
    python monitor_beta_scan.py                 # print grouped status table
    python monitor_beta_scan.py --plot          # + render trajectory PNG
    python monitor_beta_scan.py --project NAME   # override project
"""
import argparse
import json
import os
import sys
import urllib.request

ENTITY = "AIND-disRNN"
DEFAULT_PROJECT = "disrnn_updnet_bottleneck_ratio_100mice"
RUN_PREFIX = "updnet-ratio-100mice-"
N_WARMUP_DEFAULT = 7500
N_MAIN_TARGET = 150000

METRICS = [
    "train/loss",
    "bottlenecks/total_sigma",
    "bottlenecks/update_net_latent_frac_open",
    "bottlenecks/latent_frac_open",
]


def _gql(query, variables):
    key = os.environ["WANDB_API_KEY"]
    import base64

    auth = base64.b64encode(f"api:{key}".encode()).decode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
            "User-Agent": "beta-scan-monitor",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def _unwrap(v):
    return v["value"] if isinstance(v, dict) and "value" in v else v


def _flat(cfg):
    c = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
    out = {}

    def rec(d, p=""):
        for k, v in d.items():
            v2 = _unwrap(v)
            if isinstance(v2, dict):
                rec(v2, p + k + ".")
            else:
                out[p + k] = v2

    rec(c)
    return out


def fetch_runs(project):
    q = """query P($e:String!,$p:String!){project(name:$p,entityName:$e){
             runs(first:200){edges{node{name state config summaryMetrics}}}}}"""
    data = _gql(q, {"e": ENTITY, "p": project})
    edges = data["data"]["project"]["runs"]["edges"]
    runs = []
    for e in edges:
        n = e["node"]
        if not n["name"].startswith(RUN_PREFIX):
            continue
        f = _flat(n["config"])
        s = (
            json.loads(n["summaryMetrics"])
            if isinstance(n["summaryMetrics"], str)
            else (n["summaryMetrics"] or {})
        )
        base = f.get("model.penalties.beta")
        unl = f.get("model.penalties.update_net_latent_penalty")
        mult = round(unl / base) if (base and unl) else None
        runs.append(
            {
                "name": n["name"],
                "state": n["state"],
                "mult": mult,
                "beta": base,
                "lr": f.get("model.training.lr"),
                "seed": f.get("model.seed"),
                "step": s.get("_step") or 0,
                "loss": s.get("train/loss"),
                "total_sigma": s.get("bottlenecks/total_sigma"),
                "unl_frac_open": s.get("bottlenecks/update_net_latent_frac_open"),
                "lat_frac_open": s.get("bottlenecks/latent_frac_open"),
            }
        )
    return runs


def fetch_history(project, run_name, samples=500):
    q = """query H($p:String!,$e:String!,$n:String!,$specs:[JSONString!]!){
             project(name:$p,entityName:$e){run(name:$n){sampledHistory(specs:$specs)}}}"""
    spec = json.dumps({"keys": ["_step"] + METRICS, "samples": samples})
    data = _gql(q, {"p": project, "e": ENTITY, "n": run_name, "specs": [spec]})
    try:
        return data["data"]["project"]["run"]["sampledHistory"][0]
    except Exception:
        return []


def print_table(runs):
    def main_step(r):
        return max(0, (r["step"] or 0) - N_WARMUP_DEFAULT)

    print(f"\n{'=' * 78}")
    print(f"beta-scan grid: {len(runs)} runs logged")
    by_state = {}
    for r in runs:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
    print("by state:", by_state)
    print(f"{'=' * 78}")
    hdr = f"{'mult':>4} {'beta':>7} {'lr':>6} {'sd':>2} | {'state':8} {'_step':>7} {'main%':>6} | {'loss':>6} {'totσ':>6} {'unl_open':>8} {'lat_open':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(runs, key=lambda x: (x["mult"] or 0, x["beta"] or 0, x["lr"] or 0, x["seed"] or 0)):
        pct = 100.0 * main_step(r) / N_MAIN_TARGET

        def fmt(v, f="{:.3f}"):
            return f.format(v) if isinstance(v, (int, float)) else "  -  "

        flag = " <-- FAILED" if r["state"] == "failed" else ""
        print(
            f"{str(r['mult']):>4} {str(r['beta']):>7} {str(r['lr']):>6} {str(r['seed']):>2} | "
            f"{r['state']:8} {r['step']:7d} {pct:5.1f}% | "
            f"{fmt(r['loss'])} {fmt(r['total_sigma'],'{:.2f}')} {fmt(r['unl_frac_open'],'{:.2f}'):>8} {fmt(r['lat_frac_open'],'{:.2f}'):>8}{flag}"
        )
    # failures summary
    fails = [r for r in runs if r["state"] == "failed"]
    if fails:
        print(f"\n{len(fails)} FAILED (likely NaN in high-lr corner):")
        for r in fails:
            print(f"  mult={r['mult']} beta={r['beta']} lr={r['lr']} seed={r['seed']} (died at _step={r['step']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="beta_scan_monitor.png")
    args = ap.parse_args()

    runs = fetch_runs(args.project)
    if not runs:
        print(f"No runs found in {ENTITY}/{args.project}")
        return
    print_table(runs)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # runs with a real trajectory
        traj = [r for r in runs if (r["step"] or 0) > N_WARMUP_DEFAULT + 200]
        traj = sorted(traj, key=lambda x: -(x["step"] or 0))[:12]
        if not traj:
            print("\n(no run far enough into main training to plot yet)")
            return
        hist = {r["name"]: fetch_history(args.project, r["name"]) for r in traj}
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        plot_metrics = [
            ("train/loss", "Training loss", "loss (lower=better)"),
            ("bottlenecks/total_sigma", "Total bottleneck sigma", "total sigma"),
            ("bottlenecks/update_net_latent_frac_open", "Update-net latent frac open", "frac open (1=open)"),
        ]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(traj)))
        for ax, (key, title, ylab) in zip(axes, plot_metrics):
            for r, c in zip(traj, colors):
                rows = hist.get(r["name"], [])
                xy = [(rr.get("_step"), rr.get(key)) for rr in rows if rr.get("_step") is not None and rr.get(key) is not None]
                if xy:
                    x, y = zip(*xy)
                    ax.plot(x, y, color=c, lw=1.3, label=f"m{r['mult']} b{r['beta']} lr{r['lr']} s{r['seed']}")
            ax.axvline(N_WARMUP_DEFAULT, color="0.7", ls="--", lw=0.8)
            ax.set_title(title)
            ax.set_xlabel("wandb _step (warmup<7500)")
            ax.set_ylabel(ylab)
            ax.margins(0.04)
        axes[0].legend(fontsize=5.5, frameon=False)
        fig.suptitle(f"beta-scan monitor: loss vs sparsity ({len(traj)} runs)", y=1.02)
        fig.tight_layout()
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"\nsaved plot -> {args.out}")


if __name__ == "__main__":
    main()
