"""Synthetic input tests; no test writes to the project's protected data folder."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from criteo_uplift import data


def _synthetic_frame(n=400):
    row = np.arange(n)
    frame = pd.DataFrame(
        {feature: row * (j + 1.0) + j / 8.0 for j, feature in enumerate(data.FEATURES)}
    )
    frame["treatment"] = row % 2
    frame["conversion"] = (row // 2) % 2
    frame["visit"] = np.maximum(frame.conversion, row % 5 == 0).astype(int)
    frame["exposure"] = frame.treatment * (row % 3 != 0)
    return frame[data.COLUMNS]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="criteo_data_test_")
        self.root = Path(self.directory.name)
        self.source = self.root / "synthetic.csv"
        self.frame = _synthetic_frame()
        self.frame.to_csv(self.source, index=False)

    def tearDown(self):
        self.directory.cleanup()

    def test_sampling_reads_entire_file_is_reproducible_and_chunk_independent(self):
        before = data.fingerprint(self.source)
        first, audit = data.scan_and_sample(
            self.source, sample_size=43, seed=31, chunk_size=57
        )
        repeated, _ = data.scan_and_sample(
            self.source, sample_size=43, seed=31, chunk_size=101
        )
        different, _ = data.scan_and_sample(
            self.source, sample_size=43, seed=32, chunk_size=57
        )
        pd.testing.assert_frame_equal(first, repeated)
        self.assertFalse(first.row_id.equals(different.row_id))
        self.assertEqual(len(first), 43)
        self.assertEqual(first.row_id.nunique(), 43)
        self.assertTrue(first.row_id.is_monotonic_increasing)
        self.assertGreater(first.row_id.max(), 300)
        np.testing.assert_array_equal(first.f0, first.row_id)
        self.assertEqual(audit["n_rows"], 400)
        self.assertEqual(audit["sample_n"], 43)
        self.assertEqual(audit["n_treatment"], 200)
        self.assertEqual(audit["groups"]["0"]["conversion"]["positive"], 100)
        self.assertEqual(data.fingerprint(self.source), before)

    def test_full_data_audit_counts_duplicates_without_removing_rows(self):
        duplicated = pd.concat(
            [self.frame, self.frame.iloc[[7, 7, 210]]], ignore_index=True
        )
        duplicated.to_csv(self.source, index=False)
        sample, audit = data.scan_and_sample(self.source, sample_size=0, chunk_size=61)
        self.assertEqual(len(sample), len(duplicated))
        self.assertEqual(audit["duplicate_row_fingerprints"], 3)
        self.assertEqual(sample.row_id.nunique(), len(duplicated))
        self.assertEqual(audit["n_treatment"], int(duplicated.treatment.sum()))
        self.assertEqual(audit["control_exposed"], 0)
        self.assertEqual(audit["conversion_without_visit"], 0)
        for feature_stats in audit["feature_stats"]:
            column = feature_stats["feature"]
            for arm in (0, 1):
                values = duplicated.loc[duplicated.treatment == arm, column]
                self.assertAlmostEqual(
                    feature_stats[f"mean_{arm}"], values.mean(), places=9
                )
                self.assertAlmostEqual(
                    feature_stats[f"std_{arm}"], values.std(ddof=1), places=9
                )

    def test_sample_larger_than_input_retains_every_original_row(self):
        sample, _ = data.scan_and_sample(self.source, sample_size=1000, chunk_size=37)
        np.testing.assert_array_equal(sample.row_id, np.arange(len(self.frame)))

    def test_readonly_guard_rejects_direct_and_symlinked_data_destinations(self):
        protected = self.root / "protected_data"
        protected.mkdir()
        marker = protected / "existing.json"
        marker.write_text("original", encoding="utf-8")
        link = self.root / "alias"
        link.symlink_to(protected, target_is_directory=True)
        with patch.object(data, "DATA_ROOT", protected):
            for destination in (
                protected,
                marker,
                link / "existing.json",
                protected / "new" / "out.json",
            ):
                with self.subTest(destination=destination), self.assertRaises(
                    ValueError
                ):
                    data.write_json({"replace": True}, destination)
            with self.assertRaises(ValueError):
                data.save_sample(pd.DataFrame(), link / "sample.npz")
            destination = self.root / "artifacts" / "allowed.json"
            data.write_json({"ok": True}, destination)
            self.assertTrue(destination.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "original")
        self.assertFalse((protected / "new").exists())
        self.assertFalse((protected / "sample.npz").exists())

    def test_sample_round_trip_uses_an_output_directory(self):
        sample, _ = data.scan_and_sample(
            self.source, sample_size=33, seed=2, chunk_size=41
        )
        destination = self.root / "artifacts" / "sample.npz"
        data.save_sample(sample, destination)
        loaded = data.load_sample(destination)
        pd.testing.assert_frame_equal(sample, loaded)

    def test_split_is_disjoint_reproducible_and_preserves_joint_strata(self):
        frame = _synthetic_frame(2000)
        frame.insert(0, "row_id", np.arange(len(frame)))
        parts = data.split_sample(frame, outcome="conversion", seed=42)
        repeated = data.split_sample(frame, outcome="conversion", seed=42)
        expected_sizes = {"train": 1200, "validation": 400, "test": 400}
        identifiers = []
        expected_proportions = frame.groupby(["treatment", "conversion"]).size() / len(
            frame
        )
        for name, part in parts.items():
            self.assertEqual(len(part), expected_sizes[name])
            pd.testing.assert_frame_equal(part, repeated[name])
            proportions = part.groupby(["treatment", "conversion"]).size() / len(part)
            np.testing.assert_allclose(
                proportions, expected_proportions, atol=1.0 / len(part)
            )
            identifiers.append(set(part.row_id))
        self.assertFalse(identifiers[0] & identifiers[1])
        self.assertFalse(identifiers[0] & identifiers[2])
        self.assertFalse(identifiers[1] & identifiers[2])
        self.assertEqual(set.union(*identifiers), set(frame.row_id))
        visit_parts = data.split_sample(frame, outcome="visit", seed=42)
        self.assertEqual(sum(map(len, visit_parts.values())), len(frame))

    def test_invalid_schema_nonfinite_and_nonbinary_labels_fail(self):
        invalid_frames = []
        invalid_frames.append(self.frame.drop(columns="f11"))
        invalid_frames.append(self.frame.assign(extra=0))
        invalid_frames.append(self.frame[self.frame.columns[::-1]])
        nonfinite = self.frame.copy()
        nonfinite.loc[350, "f0"] = np.inf
        invalid_frames.append(nonfinite)
        missing = self.frame.copy()
        missing.loc[350, "f2"] = np.nan
        invalid_frames.append(missing)
        for label in data.LABELS:
            invalid = self.frame.copy()
            invalid.loc[350, label] = 2
            invalid_frames.append(invalid)
        invalid_frames.append(self.frame.assign(treatment=1))
        for index, frame in enumerate(invalid_frames):
            with self.subTest(invalid_case=index):
                frame.to_csv(self.source, index=False)
                # Bad rows beyond the requested sample still invalidate the scan.
                with self.assertRaises(ValueError):
                    data.scan_and_sample(self.source, sample_size=5, chunk_size=73)
        with self.assertRaises(ValueError):
            data.scan_and_sample(self.source, sample_size=-1)
        with self.assertRaises(ValueError):
            data.scan_and_sample(self.source, chunk_size=0)

    def test_split_rejects_unsupported_outcome_and_too_small_strata(self):
        with self.assertRaises(ValueError):
            data.split_sample(self.frame, outcome="exposure")
        tiny = self.frame.iloc[:4]
        with self.assertRaises(ValueError):
            data.split_sample(tiny, outcome="conversion")


if __name__ == "__main__":
    unittest.main()
