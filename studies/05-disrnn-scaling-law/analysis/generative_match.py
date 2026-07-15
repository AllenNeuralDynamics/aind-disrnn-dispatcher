#!/usr/bin/env python
"""r4 -- generative behavioral match vs D: does the disRNN BEHAVE like a mouse?

Producer for r4. Reads the 15 finished `generative-dscan` rollouts (5 D x 3 seeds), aggregates the
same two model-vs-animal curves study 01 used for the GRU, and overlays the GRU's committed numbers
so the two models are read on one axis.

Why this exists: held-out likelihood is headroom-poor -- it puts the disRNN only ~0.010 below the
GRU and merely level with a per-mouse RL baseline. A generative test can discriminate where
likelihood cannot: does the interpretable model actually *behave* like a mouse, or does it just
assign similar per-trial probabilities?

  (1) switch-triggered: p_switch(t+1) | (reward at t) x (preceding run length) -- 4 bins.
  (2) history-dependent: p_switch(t) | last N trials' (choice, reward) pattern, abstract encoding.

Metric convention is study 01's, verbatim: headline = subject-mean Pearson correlation; companion =
subject-balanced RMSE = sqrt of the subject-balanced delta MSE. Both on the `combined` partition.

GRU comparison is study 01's **v2** arm (session conditioning ACTIVE), which is the arm whose
architecture matches these disRNN runs (session_encoding_type=scalar).

Reads:  W&B group generative-dscan-mult2@20260714-060524 (project AIND-disRNN/disrnn_data_scaling)
        studies/01-gru-scaling-law/analysis/generative_match.json  (committed GRU numbers)
Writes: analysis/generative_match.json, analysis/fig_generative_match.png
        + regenerates the result markers in analysis/reports/r4-generative-behavioral-match.md
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import wandb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))                      # this study's wandb_keys
sys.path.insert(0, str(STUDY.parent / "util"))     # shared studies/util (_meta, plot_style)

from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402
from wandb_keys import hist_corr, hist_mse, switch_corr, switch_mse  # noqa: E402

PROJECT = "AIND-disRNN/disrnn_data_scaling"

# Pin the ONE good group. Two earlier launches (…@20260714-052533, …@20260714-054028) left partial
# finished runs behind; they ran BEFORE wrapper #60, so their off-curriculum sessions were simulated
# on the wrong task family. Including them would silently mix bad rollouts into the result.
GROUP = "generative-dscan-mult2@20260714-060524"

GRU_JSON = STUDY.parent / "01-gru-scaling-law" / "analysis" / "generative_match.json"
GRU_ARM = "v2"  # session conditioning active -- the arm architecturally matched to these runs

# RL baselines: rolled out generatively from variants/generative-rl-baseline (r1's per-subject
# fits, expanded to the n=614 cohort -- see that variant's notes.md for why per-subject, not
# per-session). Single point each at D=614 (fit on all 614 mice, same cohort as the D=614 disRNN/
# GRU cells), NOT a D-sweep -- unlike the disRNN/GRU curves. JSON committed under that variant's
# results/ dir since these are this study's own work product, not a W&B-logged run.
# NB: not named "results/" -- that collides with a repo-wide .gitignore pattern (meant for the
# Beaker /results output-mount convention) and silently drops the committed JSON.
RL_RESULTS_DIR = STUDY / "variants" / "generative-rl-baseline" / "rl_rollout_summaries"
RL_LABELS = {"ctt": "compare-to-threshold", "bari": "Bari", "hattori": "Hattori"}

DLAB = {0.016: 10, 0.049: 30, 0.163: 100, 0.489: 300, 1.0: 614}

STAT = "post_switch_by_reward_and_run_length"
HIST_PATTERN, HIST_N_BACK = "abstract", 3


def collect() -> list[dict]:
    api = wandb.Api()
    runs = [r for r in api.runs(PROJECT, {"group": GROUP}) if r.state == "finished"]
    by_cell: dict[tuple, object] = {}
    for r in runs:
        meta = r.config.get("meta", {}) or {}
        ratio, seed = meta.get("source_subject_ratio"), meta.get("source_seed")
        if ratio is None:
            continue
        key = (round(float(ratio), 3), seed)
        prev = by_cell.get(key)
        if prev is None or str(getattr(r, "created_at", "")) > str(getattr(prev, "created_at", "")):
            by_cell[key] = r

    rows = []
    for (ratio, seed), r in by_cell.items():
        s = r.summary
        corr, mse = s.get(switch_corr(STAT)), s.get(switch_mse(STAT))
        if corr is None or mse is None:
            print(f"    [skip {r.name[:40]}] missing combined switch scalars")
            continue
        hc = s.get(hist_corr(HIST_PATTERN, HIST_N_BACK))
        hm = s.get(hist_mse(HIST_PATTERN, HIST_N_BACK))
        rows.append(
            dict(
                D=DLAB.get(ratio, ratio),
                seed=seed,
                corr=float(corr),
                rmse=math.sqrt(float(mse)),
                hist_corr=float(hc) if hc is not None else None,
                hist_rmse=math.sqrt(float(hm)) if hm is not None else None,
            )
        )
    # All finished cells missing the scalars means a renamed key, not a few partial runs.
    if by_cell and not rows:
        raise KeyError(
            f"none of {len(by_cell)} finished cells in {GROUP} carry the switch-triggered "
            f"scalars (wrapper schema changed? see analysis/wandb_keys.py)"
        )
    print(f"  {len(runs)} finished rollouts -> {len(rows)} cells")
    return rows


def aggregate(rows: list[dict]) -> dict:
    agg: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for field in ("corr", "rmse", "hist_corr", "hist_rmse"):
            if r[field] is not None:
                agg[r["D"]][field].append(r[field])
    out = {}
    for D, m in sorted(agg.items()):
        out[f"D{D}"] = dict(
            D=D,
            n_seeds=len(m["corr"]),
            **{
                f"{field}_{stat}": float(fn(m[field]))
                for field in ("corr", "rmse", "hist_corr", "hist_rmse")
                for stat, fn in (("mean", np.mean), ("sd", np.std))
                if m[field]
            },
        )
    return out


def gru_reference() -> dict:
    """Study 01's committed GRU numbers -- same metric, same held-out cohort, same curves."""
    gru = json.loads(GRU_JSON.read_text())
    hist = gru["history_dependent"][HIST_PATTERN][str(HIST_N_BACK)]
    out = {}
    for D in (10, 30, 100, 300, 614):
        cell, hcell = gru.get(f"{GRU_ARM}_D{D}"), hist.get(f"{GRU_ARM}_D{D}")
        if not cell:
            continue
        out[f"D{D}"] = dict(
            D=D,
            corr_mean=cell["corr_mean"],
            rmse_mean=cell["rmse_mean"],
            hist_corr_mean=(hcell or {}).get("corr_mean"),
            hist_rmse_mean=(hcell or {}).get("rmse_mean"),
        )
    return out


def rl_reference() -> dict:
    """RL baselines' generative match, computed stats-only (no figures) from cached rollouts.

    quantitative_summary.json per alias has the SAME nested structure the wrapper logs to W&B
    (see wandb_keys.py) -- just not flattened into dotted keys, since these rollouts were never
    logged to a W&B run.
    """
    out = {}
    for alias, label in RL_LABELS.items():
        path = RL_RESULTS_DIR / f"{alias}_quantitative_summary.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        sw = d["switch_triggered"]["quantitative_summary"]["subject_mean"][STAT]
        sw_mse = d["switch_triggered"]["delta_significance_summary"][STAT][
            "subject_balanced_error_summary"
        ]["mean_squared_error"]
        hist = d["history_dependent"]["quantitative_summary"]["subject_mean"][HIST_PATTERN][
            str(HIST_N_BACK)
        ]
        hist_mse = d["history_dependent"]["delta_significance_summary"][HIST_PATTERN][
            str(HIST_N_BACK)
        ]["subject_balanced_error_summary"]["mean_squared_error"]
        out[alias] = dict(
            label=label,
            D=614,
            corr_mean=float(sw["correlation"]),
            rmse_mean=math.sqrt(float(sw_mse)),
            hist_corr_mean=float(hist["correlation"]),
            hist_rmse_mean=math.sqrt(float(hist_mse)),
        )
    return out


def figure(disrnn: dict, gru: dict, rl: dict) -> None:
    apply_presentation_style()
    Ds = sorted(v["D"] for v in disrnn.values())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for ax, (corr_key, rmse_key, title) in zip(
        axes,
        (
            ("corr", "rmse", "Switch-triggered curve (4 bins)"),
            ("hist_corr", "hist_rmse", f"History pattern ({HIST_PATTERN}, n={HIST_N_BACK})"),
        ),
    ):
        dm = [disrnn[f"D{D}"].get(f"{corr_key}_mean") for D in Ds]
        ds = [disrnn[f"D{D}"].get(f"{corr_key}_sd", 0.0) for D in Ds]
        gm = [gru.get(f"D{D}", {}).get(f"{corr_key}_mean") for D in Ds]
        ax.errorbar(Ds, dm, yerr=ds, fmt="o-", capsize=3, label="disRNN (mult=2, β=1e-3)")
        ax.plot(Ds, gm, "s--", color="gray", label="GRU (study 01, v2)")
        rl_markers = {"ctt": "^", "bari": "v", "hattori": "D"}
        for alias, v in rl.items():
            ax.scatter(
                [v["D"]], [v.get(f"{corr_key}_mean")],
                marker=rl_markers.get(alias, "x"), s=70, color="#c44e52", zorder=5,
                label=f"RL: {v['label']}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("# training mice (D)")
        ax.set_ylabel("model-vs-animal correlation")
        ax.set_title(title, fontsize=11)
        # upper-left is the one corner ALL series (disRNN/GRU/RL) leave clear at every D --
        # lower-right sat directly on top of the RL:ctt history-panel marker (hist_corr=0.93).
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Generative behavioral match vs cohort size — the disRNN is less mouse-like at every D",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "fig_generative_match.png", dpi=150)
    plt.close(fig)


def _fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def render_markers(disrnn: dict, gru: dict, rl: dict) -> None:
    Ds = sorted(v["D"] for v in disrnn.values())
    header = "| D | " + " | ".join(str(D) for D in Ds) + " |"
    sep = "|---|" + "---|" * len(Ds)

    def row(label, get):
        return f"| **{label}** | " + " | ".join(get(D) for D in Ds) + " |"

    lines = [
        "**(a) Switch-triggered curve** — `post_switch_by_reward_and_run_length` "
        "(subject-mean correlation, 3 seeds):",
        "",
        header,
        sep,
        row("disRNN corr", lambda D: _fmt(disrnn[f"D{D}"].get("corr_mean"))),
        row("GRU corr", lambda D: _fmt(gru.get(f"D{D}", {}).get("corr_mean"))),
        row("disRNN RMSE", lambda D: _fmt(disrnn[f"D{D}"].get("rmse_mean"))),
        row("GRU RMSE", lambda D: _fmt(gru.get(f"D{D}", {}).get("rmse_mean"))),
        "",
        f"**(b) History-pattern curve** — `history_dependent`, {HIST_PATTERN} encoding, "
        f"n_back={HIST_N_BACK} (32 bins):",
        "",
        header,
        sep,
        row("disRNN corr", lambda D: _fmt(disrnn[f"D{D}"].get("hist_corr_mean"))),
        row("GRU corr", lambda D: _fmt(gru.get(f"D{D}", {}).get("hist_corr_mean"))),
        row("disRNN RMSE", lambda D: _fmt(disrnn[f"D{D}"].get("hist_rmse_mean"))),
        row("GRU RMSE", lambda D: _fmt(gru.get(f"D{D}", {}).get("hist_rmse_mean"))),
    ]

    if rl:
        d614, g614 = disrnn.get("D614", {}), gru.get("D614", {})
        lines += [
            "",
            "**(c) RL baselines at D=614** — per-subject fits (r1), rolled out through the SAME "
            "task construction as the disRNN/GRU rollouts (not a D-sweep: one fit per mouse, all "
            "614 mice):",
            "",
            "| model | switch corr | switch RMSE | history corr | history RMSE |",
            "|---|---|---|---|---|",
            f"| **GRU** | {_fmt(g614.get('corr_mean'))} | {_fmt(g614.get('rmse_mean'))} | "
            f"{_fmt(g614.get('hist_corr_mean'))} | {_fmt(g614.get('hist_rmse_mean'))} |",
        ]
        for alias in ("hattori", "ctt", "bari"):
            v = rl.get(alias)
            if not v:
                continue
            lines.append(
                f"| {v['label']} | {_fmt(v['corr_mean'])} | {_fmt(v['rmse_mean'])} | "
                f"{_fmt(v['hist_corr_mean'])} | {_fmt(v['hist_rmse_mean'])} |"
            )
        lines.append(
            f"| **disRNN** | {_fmt(d614.get('corr_mean'))} | {_fmt(d614.get('rmse_mean'))} | "
            f"{_fmt(d614.get('hist_corr_mean'))} | {_fmt(d614.get('hist_rmse_mean'))} |"
        )
    block = "\n".join(lines)

    report = STUDY / "analysis" / "reports" / "r4-generative-behavioral-match.md"
    text = report.read_text()
    new = re.sub(
        r"(<!-- BEGIN result-4 -->\n).*?(\n<!-- END result-4 -->)",
        lambda m: m.group(1) + block + m.group(2),
        text,
        flags=re.S,
    )
    if new == text and "<!-- BEGIN result-4 -->" not in text:
        raise RuntimeError("result-4 markers not found in r4 report")
    report.write_text(new)


def main() -> None:
    rows = collect()
    disrnn = aggregate(rows)
    gru = gru_reference()
    rl = rl_reference()

    payload = {
        "_meta": build_meta("analysis/generative_match.py", [GROUP]),
        "metric": {
            "switch_stat": STAT,
            "history": {"pattern": HIST_PATTERN, "n_back": HIST_N_BACK},
            "partition": "combined",
            "gru_reference": {"source": str(GRU_JSON.relative_to(STUDY.parent.parent)), "arm": GRU_ARM},
            "rl_reference": {
                "source": str(RL_RESULTS_DIR.relative_to(STUDY.parent.parent)),
                "note": "single point at D=614, per-subject fits (r1), not a D-sweep",
            },
        },
        "disrnn": disrnn,
        "gru": gru,
        "rl": rl,
    }
    (HERE / "generative_match.json").write_text(json.dumps(payload, indent=2))

    figure(disrnn, gru, rl)
    render_markers(disrnn, gru, rl)

    print(f"\n{'D':>5} {'corr':>8} {'GRU':>8} {'rmse':>8} {'GRU':>8}")
    for D in sorted(v["D"] for v in disrnn.values()):
        d, g = disrnn[f"D{D}"], gru.get(f"D{D}", {})
        print(
            f"{D:>5} {_fmt(d.get('corr_mean')):>8} {_fmt(g.get('corr_mean')):>8} "
            f"{_fmt(d.get('rmse_mean')):>8} {_fmt(g.get('rmse_mean')):>8}"
        )


if __name__ == "__main__":
    main()
