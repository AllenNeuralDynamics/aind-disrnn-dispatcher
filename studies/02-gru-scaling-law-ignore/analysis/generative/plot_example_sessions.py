"""Plot example generated vs real foraging sessions using the standard
aind-dynamic-foraging-basic-analysis session plot."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session import (
    plot_foraging_session,
)

SRC = "hpc/8e046765-e7aa-4aab-ae93-9f5d43c7a717/out/example_sessions.json"
d = json.load(open(SRC))


def to_std(rec):
    """Standard convention: 0=left, 1=right, np.nan=ignored. reward as bool."""
    ch = np.array(
        [np.nan if (c is None or c == 2) else float(c) for c in rec["choice"]],
        dtype=float,
    )
    rw = np.array(rec["reward"], dtype=float) > 0.5
    return ch, rw


# paired (same source session) real vs model
pairs = [(d["model"][i], d["real"][i]) for i in (1, 2, 4)]
rows = len(pairs) * 2

fig = plt.figure(figsize=(15, 2.9 * rows), dpi=150)
outer = fig.add_gridspec(rows, 1, hspace=0.55)

for k, (mrec, rrec) in enumerate(pairs):
    for j, (rec, label) in enumerate(((rrec, "REAL"), (mrec, "MODEL (generated)"))):
        host_ax = fig.add_subplot(outer[k * 2 + j, 0])
        host_ax.axis("off")  # container only; function subdivides it
        ch, rw = to_std(rec)
        n = len(ch)
        # p_reward (the block reward schedule) is not recovered in this dump;
        # pass NaN so the schedule trace is blank but the choice panel renders.
        p_rew = np.full((2, n), np.nan)
        n_ign = int(np.isnan(ch).sum())
        _, (ax_cr, _ax_sched) = plot_foraging_session(
            choice_history=ch,
            reward_history=rw,
            p_reward=p_rew,
            ax=host_ax,
            plot_list=["choice", "finished"],
        )
        ax_cr.set_title(
            f"{label} — {rec['ses_idx']}   "
            f"({n} trials, {n_ign} ignored = {n_ign/n:.1%}, {int(rw.sum())} rewards)",
            fontsize=9, loc="left", pad=16,
        )
        # the function's default legend sits on top of the title; move it below-right
        leg = ax_cr.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.0, -0.02))
            leg.set_loc("upper right")
            leg.set_frame_on(False)
            for t in leg.get_texts():
                t.set_fontsize(6.5)
        # the p_reward schedule panel is blank (not recovered) — hide it
        _ax_sched.set_visible(False)

fig.suptitle(
    "Example sessions: real vs 3-way-GRU generated (D100/H256, run 8s29y3nc)\n"
    "Per-session ignore rate is highly variable (SD > mean), so these are "
    "illustrations of session structure, NOT evidence for the aggregate rate.",
    fontsize=11,
)
fig.savefig("example_sessions.png", dpi=150, bbox_inches="tight")
print("saved example_sessions.png")
for mrec, rrec in pairs:
    print(rrec["ses_idx"],
          "real_ign", sum(1 for c in rrec["choice"] if c is None),
          "model_ign", sum(1 for c in mrec["choice"] if c == 2))
