"""Producer for r3 — subject-capacity: is the subject bottleneck the transfer cap?

Reads the committed grid (analysis/grid.csv), writes analysis/subject_capacity.json and
analysis/fig_subject_capacity.png, and regenerates the <!-- BEGIN result-3 --> block.

    python analysis/subject_capacity_report.py     # offline; no WANDB_API_KEY needed

QUESTION. The disRNN sits ~0.010 below the GRU at every D and only ties the best per-mouse RL
baseline (r1). subject-capacity tests whether the per-subject bottleneck is the cause: 18 D=100
runs crossing subject_embedding_size {4, 16, 64} x subject_penalty {1e-3 (=beta, unchanged), 1e-4
(=beta/10), 0 (the GRU limit of the subject pathway -- unbottlenecked)}, 2 seeds each.

METRIC CAVEAT (study 03): openness is total_openness = Sigma(1-sigma). NEVER n_eff_open_frac.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "util"))  # shared studies/util (_meta, plot_style)
from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402

GROUP = "subject-capacity@20260713-225831"
EMBEDDINGS = (4, 16, 64)
PENALTIES = (0.001, 0.0001, 0.0)
PENALTY_LABEL = {0.001: "1e-3 (β, unchanged)", 0.0001: "1e-4 (β/10)", 0.0: "0 (GRU limit)"}

# Same reference numbers as scaling_report.py / r1, restated here so r3 stands alone.
DISRNN_D100 = 0.7174    # dscan-mult2 D=100, mult=2, beta=1e-3, embed=4 -- 3-seed mean (r1)
GRU_D100 = 0.7262       # study 01
RL_CTT = 0.7170         # compare-to-threshold, best per-mouse RL baseline (r1)
# per-seed dscan-mult2 D=100 values -- the regression-control target (NOT the 0.7188/0.7162
# figures sometimes quoted; those are dscan's seed 2 / seed 1, not seed 0 / seed 1).
DSCAN_D100_PER_SEED = {0: 0.71720, 1: 0.71624}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows():
    rows = []
    with (HERE / "grid.csv").open() as f:
        for r in csv.DictReader(f):
            if r["variant"] != "subject-capacity" or r["state"] != "finished":
                continue
            rows.append(
                dict(
                    emb=int(r["subject_embedding_size"]),
                    sp=_f(r["subject_penalty"]),
                    seed=int(r["seed"]),
                    ho=_f(r["heldout_ll"]),
                    open_upd_subj=_f(r["open_update_net_subj"]),
                    open_choice_subj=_f(r["open_choice_net_subj"]),
                )
            )
    return rows


def main() -> None:
    rows = load_rows()
    n = len(rows)
    print(f"subject-capacity: {n}/18 finished rows in grid.csv")

    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["emb"], r["sp"])
        for k in ("ho", "open_upd_subj", "open_choice_subj"):
            if r[k] is not None:
                cell[key][k].append(r[k])

    table = {}
    for emb in EMBEDDINGS:
        for sp in PENALTIES:
            m = cell.get((emb, sp), {})
            if not m.get("ho"):
                continue
            table[f"emb{emb}_sp{sp}"] = dict(
                emb=emb,
                sp=sp,
                n_seeds=len(m["ho"]),
                ho_mean=float(np.mean(m["ho"])),
                ho_sd=float(np.std(m["ho"])),
                open_upd_subj_mean=float(np.mean(m["open_upd_subj"])) if m.get("open_upd_subj") else None,
                open_choice_subj_mean=float(np.mean(m["open_choice_subj"])) if m.get("open_choice_subj") else None,
            )

    # regression control: emb=4, sp=1e-3 vs dscan-mult2 D=100, same seeds
    reg = {}
    for r in rows:
        if r["emb"] == 4 and r["sp"] == 0.001 and r["seed"] in DSCAN_D100_PER_SEED:
            target = DSCAN_D100_PER_SEED[r["seed"]]
            reg[r["seed"]] = dict(value=r["ho"], target=target, delta=r["ho"] - target)

    # best cell overall
    best_key = max(table, key=lambda k: table[k]["ho_mean"])
    best = table[best_key]

    payload = {
        "_meta": build_meta("analysis/subject_capacity_report.py", [GROUP]),
        "reference": {"disrnn_D100": DISRNN_D100, "gru_D100": GRU_D100, "rl_ctt": RL_CTT},
        "regression_control": reg,
        "table": table,
        "best_cell": {**best, "key": best_key},
    }
    (HERE / "subject_capacity.json").write_text(json.dumps(payload, indent=2))

    # --- figure: held-out LL vs subject_embedding_size, one line per penalty ---
    apply_presentation_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {0.001: "#1f77b4", 0.0001: "#ff7f0e", 0.0: "#2ca02c"}
    for sp in PENALTIES:
        xs, ys, es = [], [], []
        for emb in EMBEDDINGS:
            v = table.get(f"emb{emb}_sp{sp}")
            if v is None:
                continue
            xs.append(emb)
            ys.append(v["ho_mean"])
            es.append(v["ho_sd"])
        ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, color=colors[sp],
                    label=f"subject_penalty = {PENALTY_LABEL[sp]}")
    ax.axhline(DISRNN_D100, color="#888888", ls="--", lw=1.3)
    ax.text(4, DISRNN_D100, " disRNN D=100 baseline", va="bottom", ha="left", fontsize=9, color="#888888")
    ax.axhline(GRU_D100, color="#333333", ls=":", lw=1.3)
    ax.text(4, GRU_D100, " GRU D=100", va="bottom", ha="left", fontsize=9, color="#333333")
    ax.axhline(RL_CTT, color="#999999", ls=":", lw=1.3)
    ax.text(4, RL_CTT, " best RL baseline", va="top", ha="left", fontsize=9, color="#999999")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(EMBEDDINGS))
    ax.set_xticklabels([str(e) for e in EMBEDDINGS])
    ax.set_xlabel("subject_embedding_size")
    ax.set_ylabel("held-out likelihood")
    ax.set_title("subject-capacity: held-out LL vs\nembedding width and penalty (D=100)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig_subject_capacity.png", dpi=150)
    plt.close(fig)

    # --- regenerate r3 result marker ---
    lines = []
    lines.append("**Held-out likelihood, mean ± SD over 2 seeds:**")
    lines.append("")
    lines.append("| embed \\ penalty | 1e-3 (β, unchanged) | 1e-4 (β/10) | 0 (GRU limit) |")
    lines.append("|---|---|---|---|")
    for emb in EMBEDDINGS:
        cells = []
        for sp in PENALTIES:
            v = table.get(f"emb{emb}_sp{sp}")
            cells.append(f"{v['ho_mean']:.4f} ± {v['ho_sd']:.4f}" if v else "—")
        lines.append(f"| **{emb}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Subject-channel openness Σ(1−σ), mean over 2 seeds "
                  "(update←subject / choice←subject):**")
    lines.append("")
    lines.append("| embed \\ penalty | 1e-3 | 1e-4 | 0 |")
    lines.append("|---|---|---|---|")
    for emb in EMBEDDINGS:
        cells = []
        for sp in PENALTIES:
            v = table.get(f"emb{emb}_sp{sp}")
            if v and v["open_upd_subj_mean"] is not None:
                cells.append(f"{v['open_upd_subj_mean']:.2f} / {v['open_choice_subj_mean']:.3f}")
            else:
                cells.append("—")
        lines.append(f"| **{emb}** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"**Best cell:** embed={best['emb']}, subject_penalty={best['sp']} → "
                  f"**{best['ho_mean']:.4f}** (n={best['n_seeds']} seeds). "
                  f"disRNN D=100 baseline {DISRNN_D100:.4f}; GRU D=100 {GRU_D100:.4f}; "
                  f"gap to GRU {GRU_D100 - best['ho_mean']:+.4f} "
                  f"(baseline gap was {GRU_D100 - DISRNN_D100:+.4f}).")
    lines.append("")
    lines.append("**Regression control** (embed=4, subject_penalty=1e-3 vs dscan-mult2 D=100, "
                  "same seeds):")
    lines.append("")
    lines.append("| seed | subject-capacity | dscan-mult2 D=100 | Δ |")
    lines.append("|---|---|---|---|")
    for seed in sorted(reg):
        v = reg[seed]
        lines.append(f"| {seed} | {v['value']:.4f} | {v['target']:.4f} | {v['delta']:+.4f} |")
    block = "\n".join(lines)

    report = STUDY / "analysis" / "reports" / "r3-subject-capacity.md"
    text = report.read_text()
    import re
    new = re.sub(
        r"(<!-- BEGIN result-3 -->\n).*?(\n<!-- END result-3 -->)",
        lambda m: m.group(1) + block + m.group(2),
        text,
        flags=re.S,
    )
    if new == text and "<!-- BEGIN result-3 -->" not in text:
        raise RuntimeError("result-3 markers not found in r3 report")
    report.write_text(new)

    print(f"\nbest cell: {best_key} -> {best['ho_mean']:.4f}")
    print("regression control deltas:", {k: round(v["delta"], 4) for k, v in reg.items()})


if __name__ == "__main__":
    main()
