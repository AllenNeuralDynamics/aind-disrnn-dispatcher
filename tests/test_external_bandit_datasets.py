from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from external_bandit_datasets.schema import (  # noqa: E402
    interleaved_session_manifest,
    prefix_trial_manifest,
    validate_canonical_table,
    write_dataset,
)
from external_bandit_datasets.sources import SOURCES  # noqa: E402
from external_bandit_datasets.adapters import _finish  # noqa: E402
from external_bandit_datasets.cli import _names, run  # noqa: E402


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


class TestExternalBanditDatasets(unittest.TestCase):
    def test_empty_adapter_output_has_a_schema_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            _finish([], name="grossman", excluded_trials=0, split="sessions")

    def test_cli_rejects_unknown_dataset_cleanly(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            _names("not-a-dataset")

    def test_no_download_reports_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "without --no-download"):
                run("grossman", cache_root=Path(directory), download=False)

    def test_sources_are_the_correct_priority_three(self) -> None:
        self.assertEqual(list(SOURCES), ["grossman", "chen", "zid"])
        self.assertEqual(SOURCES["grossman"].digest_algorithm, "sha256")
        self.assertEqual(SOURCES["chen"].digest_algorithm, "sha256")
        self.assertEqual(SOURCES["zid"].digest_algorithm, "sha256")
        self.assertTrue(SOURCES["zid"].filename.endswith(".mat"))

    def test_interleaved_split_then_first_k_semantics(self) -> None:
        manifest = interleaved_session_manifest(
            _table(), dataset_id="test", species="mouse"
        )
        row = manifest["subjects"][0]
        self.assertEqual(row["adapt_session_ids"], ["s1", "s3"])
        self.assertEqual(row["test_session_ids"], ["s2", "s4"])
        self.assertEqual(row["adapt_session_ids"][:1], ["s1"])

    def test_interleaved_split_is_independent_of_input_row_order(self) -> None:
        table = _table()
        expected = interleaved_session_manifest(
            table, dataset_id="test", species="mouse"
        )
        shuffled = table.sample(frac=1.0, random_state=42).reset_index(drop=True)

        actual = interleaved_session_manifest(
            shuffled, dataset_id="test", species="mouse"
        )

        self.assertEqual(actual, expected)

    def test_prefix_manifest_preserves_one_session(self) -> None:
        table = _table(sessions=1)
        manifest = prefix_trial_manifest(
            table, dataset_id="test", species="human"
        )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            manifest["subjects"][0],
            {
                "subject_id": "a",
                "session_id": "s1",
                "adapt_prefix_trials": 2,
                "total_trials": 4,
            },
        )
        shuffled = table.sample(frac=1.0, random_state=42).reset_index(drop=True)
        self.assertEqual(
            prefix_trial_manifest(shuffled, dataset_id="test", species="human"),
            manifest,
        )

    def test_canonical_validation_rejects_noncontiguous_trials(self) -> None:
        table = _table(sessions=1)
        table.loc[1, "trial"] = 7
        with self.assertRaisesRegex(ValueError, "contiguous zero-based"):
            validate_canonical_table(table)

    def test_canonical_validation_accepts_shuffled_rows(self) -> None:
        table = _table().sample(frac=1.0, random_state=42).reset_index(drop=True)
        validate_canonical_table(table)

    def test_write_dataset_round_trip(self) -> None:
        table = _table()
        manifest = interleaved_session_manifest(
            table, dataset_id="test", species="mouse"
        )
        with tempfile.TemporaryDirectory() as directory:
            table_path, manifest_path = write_dataset(
                table, manifest, output_root=Path(directory), stem="test"
            )
            pd.testing.assert_frame_equal(pd.read_parquet(table_path), table)
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)
