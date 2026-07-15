#!/usr/bin/env python
"""Fetch per-(subject,session) embedding coordinates for 8 example subjects from the
Stage-2 session-conditioned run (vldd2pxy, N=200, mild monotonic drift), using the TRAINING CODE's own reconstruction
(utils.multisubject.compute_session_conditioned_context_dataframe) -- the same function
stage2_session_traj.py already uses for Stage 2. No reimplementation.

Run on an HPC compute node (disrnn-cpu, WANDB_API_KEY exported, wrapper importable).
Writes stage2_embedding_trajectories.csv: subject_id, session_id, session_phase,
embedding_1..4 for 8 example subjects.
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
import wandb

sys.path.insert(0, os.path.expanduser("~/code/aind-disrnn-wrapper/code"))
from utils.multisubject import compute_session_conditioned_context_dataframe

ENT, PROJ = "AIND-disRNN", "embedding_recovery"
RUN = "vldd2pxy"
N_EXAMPLES = 8

api = wandb.Api()


def fetch(fn, dest):
    a = api.artifact(f"{ENT}/{PROJ}/gru-output-{RUN}:latest", type="training-output")
    return a.get_entry(fn).download(root=dest)


if __name__ == "__main__":
    rd = "dl"
    os.makedirs(rd, exist_ok=True)
    P = json.load(open(fetch("params.json", rd)))
    md = json.load(open(fetch("multisubject_metadata.json", rd)))
    arch = json.load(open(fetch("gru_config.json", rd)))["architecture"]

    n_subj = len(md["session_max_index_by_subject_index"])
    example_indices = list(np.linspace(0, n_subj - 1, N_EXAMPLES, dtype=int))
    print("example subject indices:", example_indices)

    ctx = compute_session_conditioned_context_dataframe(
        P, session_context=md["session_context"],
        session_encoding_type=arch["session_encoding_type"],
        session_integration_type=arch.get("session_integration_type", "direct"),
        session_fourier_k=int(arch.get("session_fourier_k", 4)),
        session_delta_n_layers=int(arch["session_delta_n_layers"]),
        session_delta_hidden_size=int(arch["session_delta_hidden_size"]),
        session_curriculum_lambda=1.0,
        session_max_index_by_subject_index=md["session_max_index_by_subject_index"],
        train_session_ids=md.get("train_session_ids"), eval_session_ids=md.get("eval_session_ids"),
        selected_subject_indices=example_indices)

    keep_cols = ["subject_id", "session_id", "session_index", "subject_max_session_index",
                 "session_phase"] + [c for c in ctx.columns if c.startswith("embedding_")]
    out = ctx[keep_cols].copy()
    out.to_csv("stage2_embedding_trajectories.csv", index=False)
    print(f"WROTE stage2_embedding_trajectories.csv: {out.shape[0]} rows, "
          f"{out.subject_id.nunique()} subjects, {len([c for c in keep_cols if c.startswith('embedding_')])} embedding dims")
