"""Render the offline author-aligned baseline comparison."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style, t975  # noqa: E402


AUTHOR_DATA = STUDY / "analysis" / "author_baseline_results.json"
MATCHED_DATA = STUDY / "analysis" / "matched_half_results.json"
FIGURE = STUDY / "analysis" / "fig_author_baseline_likelihood.png"
REPORT = STUDY / "analysis" / "reports" / "r2-author-aligned-baselines.md"
START = "<!-- BEGIN result-2 -->"
END = "<!-- END result-2 -->"
DS = (10, 30, 100, 300, 614)
LABELS = {
    "grossman": "Grossman mouse",
    "chen": "Chen mouse",
    "zid": "Zid human",
}
BASELINE_LABELS = {
    "grossman-meta-learning": "meta-learning RL",
    "chen-rlck": "4-parameter RLCK",
    "zid-traditional-rlck": "traditional RLCK",
    "zid-history-kernel-foraging": "HK2 foraging RL",
}
PARAM_COUNTS = {
    "grossman-meta-learning": 7,
    "chen-rlck": 4,
    "zid-traditional-rlck": 4,
    "zid-history-kernel-foraging": 5,
}


def _metric(record: dict) -> float:
    return float(record["metrics"]["normalized_likelihood"])


def _gru_d614(dataset: dict) -> list[dict]:
    rows = [row for row in dataset["gru"] if int(row["nominal_D"]) == 614]
    if len(rows) != 3:
        raise AssertionError("Expected exactly three D=614 GRU source seeds")
    return rows


def _paired(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "low": mean - t975(len(values)) * sem,
        "high": mean + t975(len(values)) * sem,
        "fraction_positive": sum(value > 0 for value in values) / len(values),
        "n": len(values),
    }


def _paired_comparisons(record: dict, dataset: dict) -> tuple[dict, dict]:
    author = record["metrics"]["per_subject_mean_log_likelihood_nats"]
    q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
    gru = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in _gru_d614(dataset)
    ]
    if set(author) != set(q) or any(set(seed) != set(author) for seed in gru):
        raise AssertionError("Author, Q, and GRU per-subject metric sets do not align")
    author_minus_q = [float(author[key]) - float(q[key]) for key in author]
    gru_minus_author = [
        statistics.mean(float(seed[key]) for seed in gru) - float(author[key])
        for key in author
    ]
    return _paired(author_minus_q), _paired(gru_minus_author)


def _plot(author_data: dict, matched: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    records = author_data["records"]
    for axis, dataset_name in zip(axes, ("grossman", "chen", "zid")):
        dataset = matched["datasets"][dataset_name]
        baselines = [
            (key, record)
            for key, record in records.items()
            if record["dataset"] == dataset_name
        ]
        means, sds = [], []
        for d in DS:
            values = [
                _metric(row) for row in dataset["gru"] if int(row["nominal_D"]) == d
            ]
            if len(values) != 3:
                raise AssertionError(f"Expected three GRU seeds for D={d}")
            means.append(statistics.mean(values))
            sds.append(statistics.stdev(values))
            axis.scatter([d] * len(values), values, s=28, color="#4C72B0", alpha=0.4)
        axis.errorbar(
            DS,
            means,
            yerr=sds,
            marker="o",
            color="#4C72B0",
            capsize=3,
            label="GRU mean ± SD",
            zorder=4,
        )
        axis.axhline(
            _metric(dataset["q"]),
            color="#333333",
            linestyle="--",
            label="common Q",
        )
        for baseline, record in baselines:
            selected = bool(record["author_selected"])
            suffix = "author-selected" if selected else "paper comparator"
            axis.axhline(
                _metric(record),
                color="#C44E52" if selected else "#DD8452",
                linestyle="-." if selected else ":",
                label=f"{BASELINE_LABELS[baseline]} ({suffix})",
            )
        axis.set_xscale("log")
        axis.set_xticks(DS, [str(d) for d in DS])
        axis.set_title(LABELS[dataset_name])
        axis.set_xlabel("Source subjects D")
        axis.set_ylabel("Held-out normalized likelihood")
        axis.grid(axis="y", alpha=0.2)
        legend_location = "upper left" if dataset_name == "zid" else "best"
        axis.legend(frameon=False, fontsize=8.5, loc=legend_location)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def _interval(stats: dict) -> str:
    return f"{stats['mean']:+.5f} [{stats['low']:+.5f}, {stats['high']:+.5f}]"


def _result_block(author_data: dict, matched: dict) -> str:
    lines = [
        "[regenerated by `analysis/report_author_baselines.py` — do not edit by hand]",
        "",
        "![Author-aligned baselines versus common Q and transferred GRU](../fig_author_baseline_likelihood.png)",
        "",
        "All models use the same subject-level adaptation/test split and score the exact same held-out trials. "
        "The GRU curve is mean ± SD across three source-training seeds at every D; the table reports D=614.",
        "",
        "| target | published baseline | selected by authors? | parameters | common Q | published baseline | GRU D=614 |",
        "|---|---|:---:|---:|---:|---:|---:|",
    ]
    comparisons = []
    for baseline, record in author_data["records"].items():
        dataset = matched["datasets"][record["dataset"]]
        gru_values = [_metric(row) for row in _gru_d614(dataset)]
        lines.append(
            f"| {LABELS[record['dataset']]} | {BASELINE_LABELS[baseline]} | "
            f"{'yes' if record['author_selected'] else 'no — paper comparator'} | "
            f"{PARAM_COUNTS[baseline]} | {_metric(dataset['q']):.5f} | "
            f"**{_metric(record):.5f}** | "
            f"{statistics.mean(gru_values):.5f} ± {statistics.stdev(gru_values):.5f} |"
        )
        comparisons.append((baseline, record, *_paired_comparisons(record, dataset)))

    lines += [
        "",
        "### Subject-balanced paired differences",
        "",
        "Values are mean log-likelihood differences in nats/trial with 95% confidence intervals. "
        "Positive favors the model named first.",
        "",
        "| target | published baseline | author − common Q | subjects author better | GRU D=614 − author | subjects GRU better |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for baseline, record, author_q, gru_author in comparisons:
        lines.append(
            f"| {LABELS[record['dataset']]} | {BASELINE_LABELS[baseline]} | "
            f"{_interval(author_q)} | {author_q['fraction_positive']:.0%} ({author_q['n']}) | "
            f"{_interval(gru_author)} | {gru_author['fraction_positive']:.0%} ({gru_author['n']}) |"
        )

    selected = {
        record["dataset"]: (baseline, record)
        for baseline, record in author_data["records"].items()
        if record["author_selected"]
    }
    lines += ["", "### Bottom line", ""]
    for dataset_name in ("grossman", "chen", "zid"):
        baseline, record = selected[dataset_name]
        dataset = matched["datasets"][dataset_name]
        gru_mean = statistics.mean(_metric(row) for row in _gru_d614(dataset))
        lines.append(
            f"- **{LABELS[dataset_name]}:** {BASELINE_LABELS[baseline]} is "
            f"{_metric(record) - _metric(dataset['q']):+.5f} versus common Q; "
            f"D=614 GRU is {gru_mean - _metric(record):+.5f} versus that published baseline."
        )
    lines += [
        "",
        "### Reproduction confidence",
        "",
        "These ratings concern whether our implementation reproduces the authors' model dynamics, "
        "not uncertainty in the measured likelihood. They do not claim reproduction of the papers' "
        "reported fit values because our adaptation/test split is intentionally different.",
        "",
        "| baseline | confidence | evidence | remaining difference from the paper |",
        "|---|:---:|---|---|",
        "| [Grossman meta-learning RL](https://pmc.ncbi.nlm.nih.gov/articles/PMC8825708/) | **Moderate** | Published value, expected/unexpected-uncertainty, asymmetric learning-rate, forgetting, bias, and softmax equations are implemented and covered by a hand-calculated trajectory test. | The paper did not release model code and used hierarchical session-level Stan fits with mouse-level hyperparameters and the constraint `negative-rate integration > expected-uncertainty rate`. We instead fit one parameter vector per subject by differential evolution on the adaptation sessions, without that ordering constraint. This is an equation-faithful model-family benchmark, not a reproduction of the paper's Bayesian fitting pipeline. |",
        "| [Chen 4-parameter RLCK](https://elifesciences.org/articles/69748) | **High** | The published `alpha`, value inverse temperature, choice-kernel learning rate, and independent kernel inverse temperature map directly to our implementation; zero initialization, chosen-value update, full choice-kernel update, and policy are covered by a hand-calculated trajectory test. | We changed the fitting data and optimizer to the common matched-half protocol and have not reproduced the paper's fitted parameters or model-agreement figure from author outputs. |",
        "| [Zid traditional RLCK](https://www.nature.com/articles/s41467-026-75773-4) (Eq. 19) | **Very high** | Initialization, value and choice-kernel updates, two inverse temperatures, policy, and bounds were checked directly against the authors' released [`model_RLchoice.m`](https://github.com/Mariemzd/HumansForageFoRwd_paper/blob/v1.0.0/modelling_matlab/model_RLchoice.m), in addition to the equation-level test. | The authors fit all 300 main trials with 20 random-start `fminsearch` fits; we fit trials 0-149 with differential evolution and score trials 150-299. |",
        "| [Zid HK2 foraging RL](https://www.nature.com/articles/s41467-026-75773-4) (Eq. 22) | **Very high** | Exploitation value initialization at 1, reset-to-threshold only after a switch, state-history kernel, and stay policy were checked directly against the authors' released [`model_ForagingFlex.m`](https://github.com/Mariemzd/HumansForageFoRwd_paper/blob/v1.0.0/modelling_matlab/model_ForagingFlex.m) and covered by a hand-calculated trajectory test. | The same intentional matched-half and optimizer differences apply. |",
        "",
        "Overall, confidence is high for the Chen and Zid model equations and state transitions. "
        "Confidence is only moderate for Grossman because the published Bayesian hierarchy and "
        "parameter-ordering constraint are not part of this matched subject-level baseline. "
        "Accordingly, the Grossman result should be described as a reimplementation of the selected "
        "model family, not an exact reproduction of the authors' full analysis.",
        "",
        "### Verification",
        "",
        "- Each published baseline was fitted independently per subject on the adaptation half; no paper-reported likelihood was copied.",
        "- Published-baseline, common-Q, and GRU prediction files have identical ordered `(subject, session, trial, choice)` keys within each target dataset.",
        "- Grossman and Chen use odd-positioned sessions for adaptation and even-positioned sessions for test. Zid adapts on trials 0–149 and scores trials 150–299 after state-only prefix replay.",
        "- The Grossman, Chen, and selected Zid implementations follow the authors' published update equations; Zid traditional RLCK is retained as the paper's simpler comparator.",
    ]
    return "\n".join(lines)


def main() -> None:
    author_data = json.loads(AUTHOR_DATA.read_text())
    matched = json.loads(MATCHED_DATA.read_text())
    _plot(author_data, matched)
    block = _result_block(author_data, matched)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + block + "\n" + text[end_start:])
    print(f"Wrote {FIGURE} and {REPORT}")


if __name__ == "__main__":
    main()
