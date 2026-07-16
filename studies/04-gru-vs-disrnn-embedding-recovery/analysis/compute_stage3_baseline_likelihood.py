#!/usr/bin/env python
"""Stage-3 baseline fit-quality: per-subject held-out likelihood for the correctly-assigned
fixed baseline vs the mis-specified alternatives, for a fit-quality comparison against GRU
(the same story as stage1/stage2's baseline-vs-GRU panel a, but Stage 3 adds a THIRD axis:
model IDENTITY, so "baseline" here means "best-of-3-fits", not one fixed family).

WHY THIS RUNS OFF THE CLAUDE-SCIENCE SANDBOX (same reason as compute_stage2_baseline_recovery.py):
the baseline_rl_output.json files live on the Allen HPC filesystem (already confirmed present
at /home/han.hou/outputs/disrnn/wandb/run-*-<rid>/files/outputs/baseline_rl_output.json for
the 3 finished stage-3 baseline runs: Bari x548cbk7, Hattori cthtvmln, CTT yi5fncw8). Run on
hpc-code (disrnn-cpu env).

Method: for each subject, eval_likelihood is read from EACH of the 3 baseline fits. The
"correctly-assigned" likelihood is the fit matching the subject's TRUE preset (so Bari
subjects fitted by the Bari model, etc.) -- this is the fair fit-quality comparison against
GRU (which also only ever has to explain the subject's own true generative process). The
best-of-3 (argmax) is ALSO reported for continuity with s3_baseline_modelselection.csv's
model-selection story, but is a different question (identity recovery, not fit quality).

Output: s3_baseline_likelihood.csv (subject_id, true_preset, eval_likelihood_matched,
eval_likelihood_bari, eval_likelihood_hattori, eval_likelihood_ctt, selected_baseline).
"""
import json, glob
import pandas as pd

RUNIDS = {"Bari2019": "x548cbk7", "Hattori2019": "cthtvmln", "CompareToThreshold": "yi5fncw8"}
OUTBASE = "/home/han.hou/outputs/disrnn/wandb"


def load_baseline(fam, rid):
    g = glob.glob(f"{OUTBASE}/run-*-{rid}/files/outputs/baseline_rl_output.json")
    if not g:
        raise FileNotFoundError(f"no baseline_rl_output.json for {fam} ({rid})")
    d = json.load(open(g[0]))
    return {sid: rec.get("eval_likelihood") for sid, rec in d["fitted_params_per_subject"].items()}, d.get("eval_likelihood")


if __name__ == "__main__":
    per_fam = {}
    pooled = {}
    for fam, rid in RUNIDS.items():
        per_fam[fam], pooled[fam] = load_baseline(fam, rid)
        print(f"{fam} ({rid}): n={len(per_fam[fam])} subjects, pooled eval_likelihood={pooled[fam]:.4f}")

    # true preset per subject: read from the already-committed model-selection CSV if present,
    # else regenerate from a GRU run's data_cfg (same generator, deterministic).
    import os, sys
    ms_path = "s3_baseline_modelselection.csv"
    if os.path.exists(ms_path):
        truemap = pd.read_csv(ms_path).set_index("subject_id")["true_preset"].to_dict()
    else:
        sys.path.insert(0, os.path.expanduser("~/scratch/recovery-smoke/aind-disrnn-wrapper/code"))
        from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents
        inv = json.load(open(os.path.expanduser("~/scratch/recovery-smoke/stage3_inventory.json")))
        dc = next(m["data_cfg"] for m in inv.values() if m.get("state") == "finished")
        ld = HierarchicalCognitiveAgents(
            task=dc["task"], agent=dc["agent"], num_trials=dc["num_trials"],
            num_subjects=dc["num_subjects"], num_sessions_per_subject=dc["num_sessions_per_subject"],
            eval_every_n=dc.get("eval_every_n", 2), batch_size=dc.get("batch_size"),
            subject_seed_stride=dc.get("subject_seed_stride", 100000), generation_workers=1,
            seed=dc.get("seed", 42), heldout_session_mode=dc.get("heldout_session_mode", "tail"),
            heldout_frac=dc.get("heldout_frac", 0.2))
        gt = ld.groundtruth_table()
        truemap = gt.groupby("subject_id").agg(true_preset=("preset_name", "first")).to_dict()["true_preset"]

    rows = []
    for sid, truep in truemap.items():
        lls = {fam: per_fam[fam].get(sid) for fam in per_fam if per_fam[fam].get(sid) is not None}
        if not lls:
            continue
        sel = max(lls, key=lls.get)
        rows.append({
            "subject_id": sid, "true_preset": truep,
            "eval_likelihood_matched": lls.get(truep),
            "eval_likelihood_bari": lls.get("Bari2019"), "eval_likelihood_hattori": lls.get("Hattori2019"),
            "eval_likelihood_ctt": lls.get("CompareToThreshold"), "selected_baseline": sel,
        })
    out = pd.DataFrame(rows)
    acc = (out.true_preset == out.selected_baseline).mean()
    mean_matched = out.eval_likelihood_matched.mean()
    print(f"n={len(out)} model-selection acc={acc:.3f} mean matched-baseline eval_likelihood={mean_matched:.4f}")
    for fam, pll in pooled.items():
        print(f"  pooled eval_likelihood ({fam} fit on its own {sum(out.true_preset==fam)} true subjects): "
              f"{out[out.true_preset==fam].eval_likelihood_matched.mean():.4f}")
    out.to_csv("s3_baseline_likelihood.csv", index=False)
    print("WROTE s3_baseline_likelihood.csv", out.shape)
