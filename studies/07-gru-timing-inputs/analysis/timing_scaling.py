#!/usr/bin/env python
"""Regenerate study-07's figures + tables from the live W&B grid.

Re-run whenever new grid cells land; it always pulls the current state of W&B
project ``mice_rt_lick_scaling`` and rewrites every derived output. Idempotent:
running twice on an unchanged grid produces an empty git diff.

    python analysis/timing_scaling.py     # WANDB_API_KEY must be set (or ~/.netrc)

Outputs (committed):
    analysis/timing_scaling.json          curated grid + _meta provenance block
    analysis/timing_scaling.csv           flat per-(arm,D) aggregate table
    fig_block_decomposition.png           (study root) arms + RT/lick split + additivity
    fig_threeway.png                      (study root) information vs width-cost decomposition
    fig_selection_breakdown.png           (study root) why D<=30 is untrustworthy
    analysis/reports/r{1,2}-*.md          BEGIN/END blocks regenerated via update_reports.py

The five arms (data.timing_features):
    OFF   enabled=False                    no added inputs (study-01-matched baseline)
    ON    enabled, reaction_time & lick    prev [logRT, n_lick_left, n_lick_right]
    RT    enabled, reaction_time only      prev logRT
    LICK  enabled, lick_counts only        prev [n_lick_left, n_lick_right]
    SHUF  enabled, shuffle=True            ON inputs permuted within session (control)

Metric: heldout/final/eval_likelihood (held-out MOUSE) -- the ONLY headline.
within-subject checkpoint/eval_likelihood is pulled purely as a
selection-reliability diagnostic (r2).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent            # studies/07-gru-timing-inputs/analysis
STUDY = HERE.parent                                # studies/07-gru-timing-inputs
sys.path.insert(0, str(HERE))                      # wandb_keys / update_reports
sys.path.insert(0, str(STUDY.parent / "util"))     # shared _meta / plot_style
import wandb_keys as K                             # noqa: E402
import update_reports                              # noqa: E402
from _meta import build_meta                       # noqa: E402

ENTITY, PROJECT = "AIND-disRNN", "mice_rt_lick_scaling"
DS = [10, 30, 100, 300, 614]
ARMS = ["OFF", "RT", "LICK", "ON", "SHUF"]
# Colours: baseline blue, RT green, lick purple, ON orange, shuffled grey.
ACOLOR = {"OFF": "#2166ac", "RT": "#4a7c59", "LICK": "#762a83",
          "ON": "#d6604d", "SHUF": "#9e9e9e"}
ALABEL = {"OFF": "no added inputs", "RT": "+ reaction time only",
          "LICK": "+ lick counts only", "ON": "+ both", "SHUF": "shuffled control"}
# D<=30 is the overtraining / broken-selection regime (see r2): excluded from
# every effect estimate, drawn as open markers.
TRUST_D = [D for D in DS if D >= 100]


# --------------------------------------------------------------------------- #
# W&B pull
# --------------------------------------------------------------------------- #
def _gql(query, variables):
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit("WANDB_API_KEY not set (or put wandb creds in ~/.netrc).")
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + base64.b64encode(f"api:{key}".encode()).decode()},
    )
    return json.load(urllib.request.urlopen(req))


_RUNS_Q = """query R($e:String!,$p:String!,$c:String){project(name:$p,entityName:$e){
 runs(first:200, after:$c){pageInfo{hasNextPage endCursor}
 edges{node{name group state config summaryMetrics}}}}}"""


def _cfg_get(cfg, *path):
    """Read a possibly-W&B-wrapped ({'value': ...}) nested config field."""
    cur = cfg
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if isinstance(cur, dict) and "value" in cur and k != "value":
            cur = cur["value"]
    return cur


def _classify_arm(tf):
    if not tf or not tf.get("enabled"):
        return "OFF"
    if tf.get("shuffle"):
        return "SHUF"
    rt = tf.get("reaction_time", True)
    lk = tf.get("lick_counts", True)
    if rt and lk:
        return "ON"
    return "RT" if rt else "LICK"


def _nearest_D(n):
    return min(DS, key=lambda t: abs(t - n))


def fetch_runs():
    """Return (rows, groups): finished runs classified into (arm, D, seed)."""
    rows, groups, cursor = [], set(), None
    while True:
        page = _gql(_RUNS_Q, {"e": ENTITY, "p": PROJECT, "c": cursor})["data"]["project"]["runs"]
        for edge in page["edges"]:
            n = edge["node"]
            if n["state"] != "finished":
                continue
            cfg = json.loads(n["config"] or "{}")
            sm = json.loads(n["summaryMetrics"] or "{}")
            tf = _cfg_get(cfg, "data", "timing_features") or {}
            subjects = _cfg_get(cfg, "resolved_subject_ids") or []
            D = len(subjects)
            if D == 0:
                continue
            heldout = sm.get(K.HELDOUT) or sm.get(K.HELDOUT_FALLBACK)
            rows.append({
                "run": n["name"][-8:], "group": n.get("group") or "",
                "arm": _classify_arm(tf), "D": D, "Db": _nearest_D(D),
                "heldout": heldout,
                "within": sm.get(K.WITHIN), "within_train": sm.get(K.WITHIN_TRAIN),
                "sel_step": sm.get(K.SELECTED_STEP), "final_step": sm.get(K.FINAL_STEP),
            })
            if n.get("group"):
                groups.add(n["group"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    return rows, sorted(groups)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def _agg(vals):
    v = np.array([x for x in vals if isinstance(x, (int, float))], float)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            "n": int(len(v)), "values": [float(x) for x in sorted(v)]}


def build_grid(rows):
    """grid[arm][D] -> {heldout: agg, within: agg, gap: agg}."""
    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r["heldout"] is not None:
            bucket[r["arm"]][r["Db"]]["heldout"].append(r["heldout"])
        if r["within"] is not None:
            bucket[r["arm"]][r["Db"]]["within"].append(r["within"])
        if r["within"] is not None and r["within_train"] is not None:
            bucket[r["arm"]][r["Db"]]["gap"].append(r["within_train"] - r["within"])
    grid = {}
    for arm in ARMS:
        grid[arm] = {}
        for D in DS:
            cell = bucket[arm][D]
            grid[arm][str(D)] = {m: _agg(cell[m]) for m in ("heldout", "within", "gap")}
    return grid


def _mean(grid, arm, D):
    c = grid[arm][str(D)]["heldout"]
    return c["mean"] if c else None


def decomposition(grid):
    """Per-D: net (ON-OFF), information (ON-SHUF), width cost (SHUF-OFF),
    RT-only and LICK-only gains over OFF, additivity shortfall."""
    out = {}
    for D in DS:
        off, on, shuf = _mean(grid, "OFF", D), _mean(grid, "ON", D), _mean(grid, "SHUF", D)
        rt, lk = _mean(grid, "RT", D), _mean(grid, "LICK", D)
        row = {}
        if None not in (off, on):
            row["net"] = on - off
        if None not in (on, shuf):
            row["information"] = on - shuf
        if None not in (shuf, off):
            row["width_cost"] = shuf - off
        if None not in (rt, off):
            row["rt_gain"] = rt - off
        if None not in (lk, off):
            row["lick_gain"] = lk - off
        if None not in (rt, lk, on, off):
            row["parts_sum"] = (rt - off) + (lk - off)
            row["shortfall"] = (on - off) - row["parts_sum"]
        out[str(D)] = row
    return out


def selection_diag(grid, rows):
    """Per-D correlation between within-subject and held-out across seeds,
    plus each metric's seed sd -- the r2 selection-reliability diagnostic."""
    out = {}
    for D in DS:
        pairs = [(r["within"], r["heldout"]) for r in rows
                 if r["Db"] == D and r["within"] is not None and r["heldout"] is not None]
        w = np.array([p[0] for p in pairs]); h = np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(w, h)[0, 1]) if len(pairs) > 2 else None
        out[str(D)] = {
            "corr_within_heldout": corr, "n": len(pairs),
            "within_sd": float(w.std(ddof=1)) if len(w) > 1 else None,
            "heldout_sd": float(h.std(ddof=1)) if len(h) > 1 else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _logx(ax, ticks=DS):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))


def fig_block_decomposition(grid, dec, path):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0))
    # (a) all arms vs D; D>=100 solid, D<=30 open/dotted (untrustworthy)
    ax = axs[0]
    for arm in ["OFF", "RT", "LICK", "ON"]:
        m = np.array([_mean(grid, arm, D) if _mean(grid, arm, D) is not None else np.nan for D in DS])
        e = np.array([(grid[arm][str(D)]["heldout"] or {}).get("sem", np.nan) for D in DS])
        big = np.array(DS) >= 100
        ax.errorbar(np.array(DS)[big], m[big], yerr=e[big], color=ACOLOR[arm],
                    marker="o", ms=7, lw=2.4, capsize=4, zorder=3)
        ax.plot(np.array(DS)[~big], m[~big], color=ACOLOR[arm], marker="o", ms=6,
                lw=1.4, ls=":", mfc="white", mew=1.4, zorder=2)
    ax.axvspan(8, 60, color="0.6", alpha=.10, zorder=0)
    _logx(ax); ax.set_xlim(8, 900)
    ax.set_xlabel("# training mice (D)"); ax.set_ylabel("held-out mouse likelihood")
    ax.set_title("Lick counts alone recover nearly all\nof the combined benefit")
    handles = [mpl.lines.Line2D([], [], color=ACOLOR[a], marker="o", lw=2.4, label=ALABEL[a])
               for a in ["OFF", "RT", "LICK", "ON"]]
    ax.legend(handles=handles, frameon=False, fontsize=12, loc="lower right")
    # (b) RT-only vs lick-only gain, D>=100
    ax = axs[1]
    for arm, key in (("LICK", "lick_gain"), ("RT", "rt_gain")):
        v = [dec[str(D)].get(key, np.nan) * 1000 for D in TRUST_D]
        ax.plot(TRUST_D, v, color=ACOLOR[arm], marker="o", ms=7, lw=2.4, label=ALABEL[arm])
    ax.axhline(0, color="k", lw=0.9)
    _logx(ax, TRUST_D); ax.set_xlim(85, 780)
    ax.set_xlabel("# training mice (D)"); ax.set_ylabel("gain over baseline (\u00d71000)")
    ax.set_title("Licking saturates; reaction time\nkeeps growing from ~zero")
    ax.legend(frameon=False, fontsize=12, loc="center right")
    # (c) additivity at D=614
    ax = axs[2]
    x = dec[str(DS[-1])]
    bars = [("+ RT\nonly", x.get("rt_gain", 0) * 1000, ACOLOR["RT"]),
            ("+ licks\nonly", x.get("lick_gain", 0) * 1000, ACOLOR["LICK"]),
            ("sum of\nparts", x.get("parts_sum", 0) * 1000, "0.6"),
            ("+ both\n(measured)", x.get("net", 0) * 1000, ACOLOR["ON"])]
    b = ax.bar(range(4), [v for _, v, _ in bars], color=[c for _, _, c in bars], width=.62)
    for bb, (_, v, _) in zip(b, bars):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.12, f"{v:+.2f}", ha="center", fontsize=12)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for l, _, _ in bars])
    ax.set_ylabel("gain over baseline (\u00d71000)")
    ax.set_title(f"Parts slightly exceed the whole at D={DS[-1]}:\nmildly redundant, not synergistic")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_threeway(grid, dec, path):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    info = [dec[str(D)].get("information", np.nan) * 1000 for D in DS]
    width = [dec[str(D)].get("width_cost", np.nan) * 1000 for D in DS]
    net = [dec[str(D)].get("net", np.nan) * 1000 for D in DS]
    ax.plot(DS, info, color=ACOLOR["ON"], marker="o", ms=7, lw=2.4, label="information (ON\u2212shuffled)")
    ax.plot(DS, width, color=ACOLOR["LICK"], marker="s", ms=7, lw=2.4, label="width cost (shuffled\u2212OFF)")
    ax.plot(DS, net, color=ACOLOR["OFF"], marker="^", ms=7, lw=1.8, ls="--", label="net (ON\u2212OFF)")
    ax.axhline(0, color="k", lw=0.9)
    ax.axvspan(8, 60, color="0.6", alpha=.10, zorder=0)
    _logx(ax); ax.set_xlim(8, 900)
    ax.set_xlabel("# training mice (D)")
    ax.set_ylabel("\u0394 held-out LL (\u00d71000)")
    ax.set_title("The effect is real & flat; the apparent\n'growth' is a vanishing overfit tax")
    ax.legend(frameon=False, fontsize=12, loc="center right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_selection_breakdown(grid, sel, rows, path):
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0))
    corr = [sel[str(D)]["corr_within_heldout"] for D in DS]
    wsd = [sel[str(D)]["within_sd"] for D in DS]
    hsd = [sel[str(D)]["heldout_sd"] for D in DS]
    # (a) selection signal
    ax = axs[0]
    ax.plot(DS, corr, color="#2166ac", marker="o", ms=8, lw=2.4, zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls=":"); ax.axvspan(8, 60, color="#b2182b", alpha=.09, zorder=0)
    for x, y in zip(DS, corr):
        if y is not None:
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11)
    _logx(ax); ax.set_xlim(8, 900); ax.set_ylim(-0.2, 1.12)
    ax.set_xlabel("# training mice (D)")
    ax.set_ylabel("corr(within-subject, held-out)\nacross seeds")
    ax.set_title("The metric best_eval selects on stops\ntracking held-out at small D")
    # (b) seed noise vs effect
    ax = axs[1]
    ax.plot(DS, wsd, color="#762a83", marker="s", ms=7, lw=2.2, label="within-subject LL")
    ax.plot(DS, hsd, color="#2166ac", marker="o", ms=7, lw=2.2, label="held-out LL")
    eff = abs(np.nanmean([dec_v for dec_v in [
        (grid["ON"][str(D)]["heldout"]["mean"] - grid["SHUF"][str(D)]["heldout"]["mean"])
        for D in TRUST_D
        if grid["ON"][str(D)]["heldout"] and grid["SHUF"][str(D)]["heldout"]]]))
    ax.axhline(eff, color="#4a7c59", lw=1.6, ls="--")
    ax.text(9, eff * 1.12, f"information effect (+{eff:.4f})", color="#4a7c59", fontsize=11, va="bottom")
    _logx(ax); ax.set_xlim(8, 900); ax.set_yscale("log")
    ax.set_xlabel("# training mice (D)"); ax.set_ylabel("seed-to-seed sd of the metric")
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    ax.set_title("Seed noise dwarfs the effect at small D;\nfalls below it by D\u2248100")
    # (c) within vs heldout scatter at the extremes
    ax = axs[2]
    for D, c, mk in ((DS[0], "#b2182b", "o"), (DS[-1], "#2166ac", "^")):
        g = [(r["within"], r["heldout"]) for r in rows
             if r["Db"] == D and r["within"] is not None and r["heldout"] is not None]
        if g:
            ax.scatter([p[0] for p in g], [p[1] for p in g], color=c, marker=mk, s=44,
                       alpha=.8, edgecolor="white", lw=.6, zorder=3, label=f"D\u2248{D}")
    ax.set_xlabel("within-subject LL (selection signal)")
    ax.set_ylabel("held-out mouse LL (reported)")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.set_title("At small D the two are uncorrelated;\nat large D they move together")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(grid, dec, sel, path):
    cols = ["D"] + [f"{a}_mean" for a in ARMS] + [f"{a}_n" for a in ARMS] + \
           ["net", "information", "width_cost", "rt_gain", "lick_gain", "shortfall",
            "corr_within_heldout", "within_sd", "heldout_sd"]
    lines = [",".join(cols)]
    for D in DS:
        row = [str(D)]
        for a in ARMS:
            c = grid[a][str(D)]["heldout"]
            row.append(f"{c['mean']:.6f}" if c else "")
        for a in ARMS:
            c = grid[a][str(D)]["heldout"]
            row.append(str(c["n"]) if c else "0")
        d = dec[str(D)]
        for k in ("net", "information", "width_cost", "rt_gain", "lick_gain", "shortfall"):
            row.append(f"{d[k]:.6f}" if k in d else "")
        s = sel[str(D)]
        row.append(f"{s['corr_within_heldout']:.4f}" if s["corr_within_heldout"] is not None else "")
        row.append(f"{s['within_sd']:.6f}" if s["within_sd"] is not None else "")
        row.append(f"{s['heldout_sd']:.6f}" if s["heldout_sd"] is not None else "")
        lines.append(",".join(row))
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def main():
    try:
        from plot_style import apply_presentation_style
        apply_presentation_style()
    except Exception:
        pass
    rows, groups = fetch_runs()
    grid = build_grid(rows)
    dec = decomposition(grid)
    sel = selection_diag(grid, rows)

    data = {
        "_meta": build_meta("analysis/timing_scaling.py", groups, study_root=STUDY),
        "project": f"{ENTITY}/{PROJECT}",
        "arms": ARMS, "D_grid": DS, "trustworthy_D": TRUST_D,
        "metric": K.HELDOUT,
        "grid": grid, "decomposition": dec, "selection_diagnostic": sel,
        "n_runs": len(rows),
    }
    (HERE / "timing_scaling.json").write_text(json.dumps(data, indent=1) + "\n")
    write_csv(grid, dec, sel, HERE / "timing_scaling.csv")

    fig_block_decomposition(grid, dec, STUDY / "fig_block_decomposition.png")
    fig_threeway(grid, dec, STUDY / "fig_threeway.png")
    fig_selection_breakdown(grid, sel, rows, STUDY / "fig_selection_breakdown.png")

    update_reports.run(data)
    print(f"timing_scaling: {len(rows)} runs, {len(groups)} groups, "
          f"wrote json/csv + 3 figures + reports")


if __name__ == "__main__":
    main()
