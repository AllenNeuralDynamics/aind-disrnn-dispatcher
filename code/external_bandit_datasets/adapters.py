"""Source-specific adapters into the shared trial-level bandit schema."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .schema import (
    interleaved_session_manifest,
    prefix_trial_manifest,
    validate_canonical_table,
)
from .sources import SOURCES, verify_source_file


AdapterResult = tuple[pd.DataFrame, dict[str, object], dict[str, object]]

EXPECTED_AUDITS: dict[str, dict[str, int]] = {
    "grossman": {
        "num_subjects": 48,
        "num_sessions": 754,
        "num_trials": 210159,
        "excluded_trials": 11456,
    },
    "chen": {
        "num_subjects": 32,
        "num_sessions": 256,
        "num_trials": 70778,
        "excluded_trials": 0,
    },
    "zid": {
        "num_subjects": 258,
        "num_sessions": 258,
        "num_trials": 77400,
        "excluded_trials": 6450,
    },
}


def _finish(
    rows: list[dict[str, object]],
    *,
    name: str,
    excluded_trials: int,
    split: str,
) -> AdapterResult:
    source = SOURCES[name]
    df = pd.DataFrame.from_records(rows)
    df = df[
        [
            "subject_id",
            "ses_idx",
            "trial",
            "animal_response",
            "rewarded",
            "earned_reward",
            *[
                column
                for column in df.columns
                if column
                not in {
                    "subject_id",
                    "ses_idx",
                    "trial",
                    "animal_response",
                    "rewarded",
                    "earned_reward",
                }
            ],
        ]
    ]
    validate_canonical_table(df)
    if split == "sessions":
        manifest = interleaved_session_manifest(
            df,
            dataset_id=source.dataset_id,
            species=source.species,
        )
    elif split == "trials":
        manifest = prefix_trial_manifest(
            df,
            dataset_id=source.dataset_id,
            species=source.species,
        )
    else:
        raise ValueError(f"Unknown split mode {split!r}.")
    session_counts = df.groupby("subject_id", sort=False)["ses_idx"].nunique()
    trial_counts = df.groupby(["subject_id", "ses_idx"], sort=False).size()
    audit = {
        "dataset_id": source.dataset_id,
        "species": source.species,
        "source_doi": source.doi,
        "source_version": source.version,
        "source_license": source.license,
        "num_subjects": int(df["subject_id"].nunique()),
        "num_sessions": int(df.groupby(["subject_id", "ses_idx"]).ngroups),
        "num_trials": int(len(df)),
        "excluded_trials": int(excluded_trials),
        "min_sessions_per_subject": int(session_counts.min()),
        "max_sessions_per_subject": int(session_counts.max()),
        "min_trials_per_session": int(trial_counts.min()),
        "max_trials_per_session": int(trial_counts.max()),
        "choice_counts": {
            str(key): int(value)
            for key, value in df["animal_response"].value_counts().sort_index().items()
        },
        "reward_counts": {
            str(key): int(value)
            for key, value in df["rewarded"].value_counts().sort_index().items()
        },
        "split_strategy": manifest["split_strategy"],
    }
    mismatches = {
        field: {"expected": expected, "actual": audit[field]}
        for field, expected in EXPECTED_AUDITS[name].items()
        if audit[field] != expected
    }
    if mismatches:
        raise ValueError(f"{name} source audit did not match the pinned release: {mismatches}.")
    return df, manifest, audit


def adapt_grossman(path: str | Path) -> AdapterResult:
    """Adapt the 48-mouse dynamic-foraging behavior cohort from MATLAB."""
    source = SOURCES["grossman"]
    path = Path(path)
    verify_source_file(path, source)
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("The Grossman adapter requires scipy.") from exc

    with zipfile.ZipFile(path) as archive:
        mat_bytes = archive.read("data/data.mat")
    behavior = loadmat(io.BytesIO(mat_bytes), simplify_cells=True)["data"][
        "dynamicForaging"
    ]["behavior"]

    rows: list[dict[str, object]] = []
    excluded_trials = 0
    for subject_id in sorted(behavior):
        for session_id in sorted(behavior[subject_id]):
            canonical_trial = 0
            for source_trial, record in enumerate(behavior[subject_id][session_id]):
                if record.get("trialType") != "CSplus":
                    excluded_trials += 1
                    continue
                left_observed = pd.notna(record.get("rewardL"))
                right_observed = pd.notna(record.get("rewardR"))
                if left_observed == right_observed:
                    excluded_trials += 1
                    continue
                choice = 0 if left_observed else 1
                reward = record["rewardL"] if left_observed else record["rewardR"]
                if reward not in (0, 1):
                    excluded_trials += 1
                    continue
                row: dict[str, object] = {
                    "subject_id": subject_id,
                    "ses_idx": session_id,
                    "trial": canonical_trial,
                    "animal_response": choice,
                    "rewarded": int(reward),
                    "earned_reward": int(reward),
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_trial": source_trial,
                    "source_trial_type": record["trialType"],
                }
                if "rewardProbL" in record and "rewardProbR" in record:
                    row["reward_probability_arm_0"] = float(record["rewardProbL"]) / 100.0
                    row["reward_probability_arm_1"] = float(record["rewardProbR"]) / 100.0
                rows.append(row)
                canonical_trial += 1
    return _finish(
        rows,
        name="grossman",
        excluded_trials=excluded_trials,
        split="sessions",
    )


_CHEN_FILE = re.compile(
    r"^cleaned up restless final data/session(?P<session>\d+)/(?P<subject>\d+)\.csv$"
)


def adapt_chen(path: str | Path) -> AdapterResult:
    """Adapt the 32-mouse, eight-session restless-bandit cohort."""
    source = SOURCES["chen"]
    path = Path(path)
    verify_source_file(path, source)
    chunks: dict[tuple[int, int], pd.DataFrame] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            match = _CHEN_FILE.match(member.filename)
            if match is None:
                continue
            key = (int(match.group("subject")), int(match.group("session")))
            with archive.open(member) as stream:
                chunks[key] = pd.read_csv(stream)

    rows: list[dict[str, object]] = []
    for (subject, session), source_df in sorted(chunks.items()):
        for trial, record in source_df.reset_index(drop=True).iterrows():
            choice = int(record["choice"]) - 1
            reward = int(record["reward"])
            rows.append(
                {
                    "subject_id": f"mouse-{subject:02d}",
                    "ses_idx": f"session-{session:02d}",
                    "trial": trial,
                    "animal_response": choice,
                    "rewarded": reward,
                    "earned_reward": reward,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "sex": "male" if subject <= 16 else "female",
                    "source_subject": subject,
                    "source_session": session,
                    "source_trial": int(record.iloc[0]),
                    "reward_probability_arm_0": float(record["left"]),
                    "reward_probability_arm_1": float(record["right"]),
                    "source_hmm_state": int(record["state"]),
                    "response_time_s": float(record["RT"]),
                }
            )
    return _finish(rows, name="chen", excluded_trials=0, split="sessions")


def adapt_zid(path: str | Path) -> AdapterResult:
    """Adapt Experiment 1, removing its fixed 25-trial practice block."""
    source = SOURCES["zid"]
    path = Path(path)
    verify_source_file(path, source)
    source_subjects = pd.read_pickle(path)
    if not isinstance(source_subjects, dict):
        raise ValueError("Expected Zid all_sub_2ab.pickle to contain a subject dictionary.")

    rows: list[dict[str, object]] = []
    practice_trials = 25
    for subject in sorted(source_subjects):
        source_df = source_subjects[subject]
        if len(source_df) != 325:
            raise ValueError(f"Expected 325 Zid trials for subject {subject}; got {len(source_df)}.")
        main_df = source_df.iloc[practice_trials:].reset_index(drop=False)
        for trial, record in main_df.iterrows():
            choice = int(record["choice"])
            reward = int(record["reward"])
            rows.append(
                {
                    "subject_id": f"human-{int(subject):03d}",
                    "ses_idx": "main",
                    "trial": trial,
                    "animal_response": choice,
                    "rewarded": reward,
                    "earned_reward": reward,
                    "dataset_id": source.dataset_id,
                    "species": source.species,
                    "source_subject": int(subject),
                    "source_trial": int(record["index"]),
                    "reward_probability_arm_0": float(record["arm1"]),
                    "reward_probability_arm_1": float(record["arm2"]),
                }
            )
    return _finish(
        rows,
        name="zid",
        excluded_trials=len(source_subjects) * practice_trials,
        split="trials",
    )


ADAPTERS: dict[str, Callable[[str | Path], AdapterResult]] = {
    "grossman": adapt_grossman,
    "chen": adapt_chen,
    "zid": adapt_zid,
}


def build_dataset(name: str, source_path: str | Path) -> AdapterResult:
    return ADAPTERS[name](source_path)
