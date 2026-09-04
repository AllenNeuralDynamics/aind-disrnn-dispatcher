from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from external_bandit_datasets.schema import (  # noqa: E402
    interleaved_session_manifest,
    prefix_trial_manifest,
    validate_canonical_table,
    write_dataset,
)
from external_bandit_datasets.sources import SOURCES  # noqa: E402


def _table(*, sessions: int = 4, trials: int = 4) -> pd.DataFrame:
    rows = []
    for subject in ("a", "b"):
        for session in range(sessions):
            for trial in range(trials):
                rows.append(
                    {
                        "subject_id": subject,
                        "ses_idx": f"s{session + 1}",
                        "trial": trial,
                        "animal_response": trial % 2,
                        "rewarded": (trial + 1) % 2,
                        "earned_reward": (trial + 1) % 2,
                    }
                )
    return pd.DataFrame(rows)


def test_sources_are_the_correct_priority_three() -> None:
    assert list(SOURCES) == ["grossman", "chen", "zid"]
    assert SOURCES["grossman"].digest_algorithm == "sha256"
    assert SOURCES["chen"].digest_algorithm == "sha256"
    assert SOURCES["zid"].digest_algorithm == "md5"


def test_interleaved_split_then_first_k_semantics() -> None:
    manifest = interleaved_session_manifest(
        _table(), dataset_id="test", species="mouse"
    )
    row = manifest["subjects"][0]
    assert row["adapt_session_ids"] == ["s1", "s3"]
    assert row["test_session_ids"] == ["s2", "s4"]
    assert row["adapt_session_ids"][:1] == ["s1"]


def test_prefix_manifest_preserves_one_session() -> None:
    manifest = prefix_trial_manifest(
        _table(sessions=1), dataset_id="test", species="human"
    )
    assert manifest["schema_version"] == 2
    assert manifest["subjects"][0] == {
        "subject_id": "a",
        "session_id": "s1",
        "adapt_prefix_trials": 2,
        "total_trials": 4,
    }


def test_canonical_validation_rejects_noncontiguous_trials() -> None:
    table = _table(sessions=1)
    table.loc[1, "trial"] = 7
    with pytest.raises(ValueError, match="contiguous zero-based"):
        validate_canonical_table(table)


def test_write_dataset_round_trip(tmp_path: Path) -> None:
    table = _table()
    manifest = interleaved_session_manifest(
        table, dataset_id="test", species="mouse"
    )
    table_path, manifest_path = write_dataset(
        table, manifest, output_root=tmp_path, stem="test"
    )
    pd.testing.assert_frame_equal(pd.read_pickle(table_path), table)
    assert json.loads(manifest_path.read_text()) == manifest
