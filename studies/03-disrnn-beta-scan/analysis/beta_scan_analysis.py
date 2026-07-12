#!/usr/bin/env python
"""beta-scan / updnet-ratio-100mice analysis.

Pulls every run in the ``disrnn_updnet_bottleneck_ratio_100mice`` W&B project and scores the
sparsity-vs-fit tradeoff of Kevin's update-net-ratio idea across
``update_net_latent_penalty_multiplier`` x base beta x lr.

Per run it extracts:
  * knobs      : update_net_latent_penalty_multiplier, base beta, lr, seed
  * fit        : ``likelihood`` (test/held-in eval likelihood, wandb.summary)
  * sparsity   : ``final/bottlenecks/*`` (added by wrapper commit 5cb7154), esp.
                 ``final/bottlenecks/update_net_latent_frac_open`` — fraction of
                 interaction (update-net latent) bottlenecks with sigma < 0.1.
                 Lower = sparser interaction = the thing the multiplier should buy.

Outputs (written next to this script):
  * beta_scan_results.csv         — one row per run (knobs + fit + all sparsity scalars)
  * fig_sparsity_vs_multiplier.png — interaction sparsity vs multiplier, faceted by lr
  * fig_likelihood_vs_multiplier.png — test likelihood vs multiplier, faceted by lr
  * fig_tradeoff.png              — sparsity vs likelihood scatter (color=multiplier, marker=lr)
  * beta_scan_verdict.md          — auto-written summary of the best ratio per lr

W&B ACCESS: the ``wandb`` SDK's ``wandb.Api()`` FAILS in the Claude Science sandbox
(spawns a wandb-core service subprocess it can't reach). This script therefore hits the
GraphQL endpoint directly with ``requests`` (auth=('api', WANDB_API_KEY)); that route is
network-allowlisted. On the HPC login node (authenticated as houhan) the ``wandb`` SDK
works too — set ``USE_SDK=1`` to use it there.

Run BEFORE launch: it degrades gracefully to an illustrative placeholder figure when the
project has no runs yet, so the plotting path is proven ahead of real data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENTITY = "AIND-disRNN"
PROJECT = "disrnn_updnet_bottleneck_ratio_100mice"
OUTDIR = Path(__file__).parent

# Primary readouts.
SPARSITY_KEY = "final/bottlenecks/update_net_latent_frac_open"  # lower = sparser interaction
LIKELIHOOD_KEY = "likelihood"  # test/held-in eval likelihood (wandb.summary)

# All per-family sparsity scalars we log, pulled into the CSV for completeness.
BOTTLENECK_FAMILIES = [
    "latent",
    "update_net_subj",
    "update_net_obs",
    "update_net_latent",
    "choice_net_subj",
    "choice_net_latent",
]

GRAPHQL_URL = "https://api.wandb.ai/graphql"

_RUNS_QUERY = """
query Runs($entity: String!, $project: String!, $cursor: String) {
  project(name: $project, entityName: $entity) {
    runs(first: 200, after: $cursor) {
      edges {
        node { name displayName state config summaryMetrics }
        cursor
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


def _wandb_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit(
            "WANDB_API_KEY not set. In the sandbox it is injected; on HPC run "
            "`wandb login` first (or export the key)."
        )
    return key


def _unwrap(v):
    """W&B config values are wrapped as {'value': x}; unwrap recursively."""
    if isinstance(v, dict) and set(v.keys()) == {"value"}:
        return _unwrap(v["value"])
    if isinstance(v, dict):
        return {k: _unwrap(x) for k, x in v.items()}
    return v


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def fetch_runs_graphql() -> list[dict]:
    """Fetch all runs of the project via GraphQL (sandbox-safe route)."""
    import requests

    key = _wandb_key()
    runs, cursor = [], None
    while True:
        r = requests.post(
            GRAPHQL_URL,
            auth=("api", key),
            json={
                "query": _RUNS_QUERY,
                "variables": {"entity": ENTITY, "project": PROJECT, "cursor": cursor},
            },
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        proj = (payload.get("data") or {}).get("project")
        if not proj:
            # project may not exist yet (pre-launch) -> no runs
            return []
        conn = proj["runs"]
        for edge in conn["edges"]:
            runs.append(edge["node"])
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    return runs


def fetch_runs_sdk() -> list[dict]:
    """Fetch via the wandb SDK (works on HPC login node)."""
    import wandb

    api = wandb.Api()
    out = []
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        out.append(
            {
                "name": run.name,
                "displayName": run.name,
                "state": run.state,
                "config": {k: {"value": v} for k, v in run.config.items()},
                "summaryMetrics": dict(run.summary),
            }
        )
    return out


def parse_runs(runs: list[dict]):
    """Return list of per-run dicts: knobs + fit + sparsity scalars."""
    rows = []
    for run in runs:
        cfg_raw = run.get("config")
        summ_raw = run.get("summaryMetrics")
        cfg = _flatten(_unwrap(json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})))
        summ = json.loads(summ_raw) if isinstance(summ_raw, str) else (summ_raw or {})

        def cfg_get(*names):
            for n in names:
                if n in cfg and cfg[n] is not None:
                    return cfg[n]
            return None

        row = {
            "name": run.get("displayName") or run.get("name"),
            "state": run.get("state"),
            "multiplier": cfg_get(
                "model.penalties.update_net_latent_penalty_multiplier",
                "penalties.update_net_latent_penalty_multiplier",
            ),
            "beta": cfg_get("model.penalties.beta", "penalties.beta"),
            "lr": cfg_get("model.training.lr", "training.lr"),
            "seed": cfg_get("seed"),
            "likelihood": summ.get(LIKELIHOOD_KEY),
            "sparsity_frac_open": summ.get(SPARSITY_KEY),
            "total_sigma": summ.get("final/bottlenecks/total_sigma"),
        }
        for fam in BOTTLENECK_FAMILIES:
            for stat in ("mean_sigma", "n_open", "n_closed", "frac_open"):
                k = f"final/bottlenecks/{fam}_{stat}"
                row[f"{fam}_{stat}"] = summ.get(k)
        rows.append(row)
    return rows


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def write_csv(rows, path):
    import csv

    if not rows:
        # still write a header-only CSV so downstream tooling has the schema
        rows = [
            {
                k: None
                for k in [
                    "name",
                    "state",
                    "multiplier",
                    "beta",
                    "lr",
                    "seed",
                    "likelihood",
                    "sparsity_frac_open",
                    "total_sigma",
                ]
            }
        ]
        rows = []  # emit header from a canonical fieldset below
    fieldnames = [
        "name",
        "state",
        "multiplier",
        "beta",
        "lr",
        "seed",
        "likelihood",
        "sparsity_frac_open",
        "total_sigma",
    ] + [f"{fam}_{stat}" for fam in BOTTLENECK_FAMILIES for stat in ("mean_sigma", "n_open", "n_closed", "frac_open")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
MULT_ORDER = [1, 2, 5, 10]


def _placeholder_note(ax, msg):
    ax.text(
        0.5,
        0.5,
        msg,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="0.35",
        style="italic",
        wrap=True,
    )


def plot_sparsity_vs_multiplier(rows, path):
    lrs = sorted({_num(r["lr"]) for r in rows if r["lr"] is not None}) or [1e-3, 5e-3]
    fig, axes = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 4.2), sharey=True, squeeze=False)
    for ax, lr in zip(axes[0], lrs):
        sub = [r for r in rows if _num(r["lr"]) == lr and r["sparsity_frac_open"] is not None]
        betas = sorted({_num(r["beta"]) for r in sub})
        for b in betas:
            pts = [(_num(r["multiplier"]), _num(r["sparsity_frac_open"])) for r in sub if _num(r["beta"]) == b]
            pts = sorted(p for p in pts if not np.isnan(p[0]) and not np.isnan(p[1]))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", label=f"β={b:g}")
        ax.set_xscale("log")
        ax.set_xticks(MULT_ORDER)
        ax.set_xticklabels(MULT_ORDER)
        ax.set_xlabel("update_net_latent_penalty_multiplier")
        ax.set_title(f"lr = {lr:g}")
        if not sub:
            _placeholder_note(ax, "no runs yet\n(placeholder — launch to populate)")
        if betas:
            ax.legend(title="base β", fontsize=8)
    axes[0][0].set_ylabel("interaction bottleneck frac_open\n(lower = sparser)")
    fig.suptitle("Does a harder update-net ratio sparsify the interaction bottleneck? (100 mice)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_likelihood_vs_multiplier(rows, path):
    lrs = sorted({_num(r["lr"]) for r in rows if r["lr"] is not None}) or [1e-3, 5e-3]
    fig, axes = plt.subplots(1, len(lrs), figsize=(5 * len(lrs), 4.2), sharey=True, squeeze=False)
    for ax, lr in zip(axes[0], lrs):
        sub = [r for r in rows if _num(r["lr"]) == lr and r["likelihood"] is not None]
        betas = sorted({_num(r["beta"]) for r in sub})
        for b in betas:
            pts = [(_num(r["multiplier"]), _num(r["likelihood"])) for r in sub if _num(r["beta"]) == b]
            pts = sorted(p for p in pts if not np.isnan(p[0]) and not np.isnan(p[1]))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "s-", label=f"β={b:g}")
        ax.set_xscale("log")
        ax.set_xticks(MULT_ORDER)
        ax.set_xticklabels(MULT_ORDER)
        ax.set_xlabel("update_net_latent_penalty_multiplier")
        ax.set_title(f"lr = {lr:g}")
        if not sub:
            _placeholder_note(ax, "no runs yet\n(placeholder — launch to populate)")
        if betas:
            ax.legend(title="base β", fontsize=8)
    axes[0][0].set_ylabel("test likelihood (higher = better fit)")
    fig.suptitle("Fit cost of squeezing the interaction net (100 mice)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(rows, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    have = [
        r
        for r in rows
        if r["sparsity_frac_open"] is not None and r["likelihood"] is not None
    ]
    if not have:
        _placeholder_note(ax, "no runs yet — the tradeoff scatter populates after launch.\n"
                              "x = interaction frac_open (←sparser), y = test likelihood (↑better).\n"
                              "Best ratio = upper-left: sparse interaction, no fit loss.")
        ax.set_xlabel("interaction bottleneck frac_open (← sparser)")
        ax.set_ylabel("test likelihood (↑ better fit)")
        ax.set_title("Sparsity–fit tradeoff across update-net ratio")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    mults = sorted({_num(r["multiplier"]) for r in have})
    cmap = plt.get_cmap("viridis")
    cnorm = {m: cmap(i / max(1, len(mults) - 1)) for i, m in enumerate(mults)}
    markers = {1e-3: "o", 5e-3: "^"}
    for r in have:
        m = _num(r["multiplier"])
        lr = _num(r["lr"])
        ax.scatter(
            _num(r["sparsity_frac_open"]),
            _num(r["likelihood"]),
            color=cnorm.get(m, "0.5"),
            marker=markers.get(lr, "s"),
            s=70,
            edgecolor="k",
            linewidth=0.5,
        )
    # legends: color = multiplier, marker = lr
    from matplotlib.lines import Line2D

    color_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cnorm[m], markeredgecolor="k", markersize=9, label=f"×{m:g}")
        for m in mults
    ]
    marker_handles = [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor="0.6", markeredgecolor="k", markersize=9, label=f"lr={lr:g}")
        for lr, mk in markers.items()
    ]
    leg1 = ax.legend(handles=color_handles, title="multiplier", loc="lower left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="lr", loc="lower right", fontsize=8)
    ax.set_xlabel("interaction bottleneck frac_open (← sparser)")
    ax.set_ylabel("test likelihood (↑ better fit)")
    ax.set_title("Sparsity–fit tradeoff across update-net ratio (100 mice)\nbest = upper-left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_verdict(rows, path):
    have = [r for r in rows if r["sparsity_frac_open"] is not None and r["likelihood"] is not None]
    lines = ["# beta-scan verdict — updnet-ratio-100mice", ""]
    if not have:
        lines += [
            "_No runs found in `disrnn_updnet_bottleneck_ratio_100mice` yet. Launch the variant, then re-run this script._",
            "",
            "Primary readout once populated: **`final/bottlenecks/update_net_latent_frac_open`** "
            "(interaction sparsity, lower = sparser) vs `update_net_latent_penalty_multiplier`, "
            "and whether test `likelihood` holds as the multiplier rises.",
        ]
        Path(path).write_text("\n".join(lines))
        return
    # best ratio per lr: sparsest interaction whose likelihood is within 1% of the lr's best
    lines += ["| lr | best multiplier (sparse, no fit loss) | frac_open | likelihood |", "|---|---|---|---|"]
    for lr in sorted({_num(r["lr"]) for r in have}):
        sub = [r for r in have if _num(r["lr"]) == lr]
        best_ll = max(_num(r["likelihood"]) for r in sub)
        ok = [r for r in sub if _num(r["likelihood"]) >= best_ll - 0.01 * abs(best_ll)]
        pick = min(ok, key=lambda r: _num(r["sparsity_frac_open"]))
        lines.append(
            f"| {lr:g} | ×{_num(pick['multiplier']):g} | {_num(pick['sparsity_frac_open']):.3f} | {_num(pick['likelihood']):.4f} |"
        )
    Path(path).write_text("\n".join(lines))


def main():
    use_sdk = os.environ.get("USE_SDK") == "1"
    runs = fetch_runs_sdk() if use_sdk else fetch_runs_graphql()
    rows = parse_runs(runs)
    print(f"[beta-scan] fetched {len(runs)} runs; {sum(1 for r in rows if r['likelihood'] is not None)} with likelihood")

    write_csv(rows, OUTDIR / "beta_scan_results.csv")
    plot_sparsity_vs_multiplier(rows, OUTDIR / "fig_sparsity_vs_multiplier.png")
    plot_likelihood_vs_multiplier(rows, OUTDIR / "fig_likelihood_vs_multiplier.png")
    plot_tradeoff(rows, OUTDIR / "fig_tradeoff.png")
    write_verdict(rows, OUTDIR / "beta_scan_verdict.md")
    print(f"[beta-scan] wrote results + 3 figures + verdict to {OUTDIR}")


if __name__ == "__main__":
    main()
