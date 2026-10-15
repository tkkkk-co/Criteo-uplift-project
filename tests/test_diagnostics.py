"""Numerical and leakage guard checks for the diagnostic pipeline."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from criteo_uplift.diagnostics import (
    c2st,
    covariate_balance,
    learnability,
    srm_summary,
)


class DiagnosticsTests(unittest.TestCase):
    def test_smd_matches_pooled_standard_deviation(self):
        frame = pd.DataFrame(
            {
                "f0": [1.0, 3.0, 2.0, 4.0],
                "f1": [7.0, 7.0, 7.0, 7.0],
                "treatment": [0, 0, 1, 1],
            }
        )
        original = frame.copy(deep=True)
        result = covariate_balance(frame, ["f0", "f1"]).set_index("feature")
        self.assertAlmostEqual(result.loc["f0", "smd"], 1 / np.sqrt(2))
        self.assertEqual(result.loc["f1", "smd"], 0.0)
        self.assertEqual(result.loc["f1", "ks_statistic"], 0.0)
        assert_frame_equal(frame, original)

    def test_post_treatment_features_are_rejected(self):
        frame = pd.DataFrame({"exposure": [0, 0, 1, 1], "treatment": [0, 0, 1, 1]})
        with self.assertRaisesRegex(ValueError, "pre-treatment"):
            covariate_balance(frame, ["exposure"])

    def test_srm_requires_external_design_ratio(self):
        skipped = srm_summary(850, 150)
        self.assertEqual(skipped["status"], "skipped")
        self.assertIsNone(skipped["pvalue"])
        self.assertIn("cannot be its own", skipped["reason"])
        balanced = srm_summary(50, 50, expected_ratio=0.5)
        self.assertEqual(balanced["chi2"], 0.0)
        self.assertEqual(balanced["pvalue"], 1.0)
        self.assertLess(srm_summary(850, 150, expected_ratio=0.5)["pvalue"], 1e-50)

    def test_c2st_complete_refits_and_permutation_pvalues(self):
        rng = np.random.default_rng(2026)
        t = np.tile([0, 1], 150)
        frame = pd.DataFrame({"treatment": t, "f0": 4 * t + rng.normal(size=len(t))})
        original = frame.copy(deep=True)
        from criteo_uplift import diagnostics

        factory = diagnostics._model
        with patch.object(diagnostics, "_model", wraps=factory) as wrapped:
            result = c2st(frame, ["f0"], sample_size=300, permutations=3, seed=7)
        self.assertEqual(wrapped.call_count, 2 * (1 + 3))
        self.assertEqual(result["train_n"] + result["test_n"], 300)
        self.assertEqual(len(result["null_auc"]), 6)
        for row in result["summary"].itertuples():
            null = result["null_auc"].query("model == @row.model")["roc_auc"]
            expected = (1 + (null >= row.roc_auc).sum()) / 4
            self.assertEqual(row.pvalue, expected)
            self.assertGreaterEqual(row.pvalue_holm, row.pvalue)
            self.assertGreater(row.roc_auc, 0.90)
        assert_frame_equal(frame, original)

    def test_learnability_dummy_uses_training_rate(self):
        train = pd.DataFrame(
            {
                "f0": np.arange(100, dtype=float),
                "treatment": np.tile([0, 1], 50),
                "conversion": np.r_[np.ones(20), np.zeros(80)],
            }
        )
        val = pd.DataFrame(
            {
                "f0": np.arange(20, dtype=float),
                "treatment": np.tile([0, 1], 10),
                "conversion": np.tile([0, 1], 10),
            }
        )
        result = learnability(train, val, ["f0"], "conversion", by_arm=False)
        dummy = result.set_index("model").loc["dummy"]
        self.assertAlmostEqual(dummy["log_loss"], -(np.log(0.2) + np.log(0.8)) / 2)
        self.assertAlmostEqual(dummy["brier"], (0.2**2 + 0.8**2) / 2)
        self.assertEqual(dummy["average_precision"], 0.5)

    def test_single_class_outcomes_are_explicit(self):
        frame = pd.DataFrame(
            {
                "f0": np.arange(20, dtype=float),
                "treatment": np.tile([0, 1], 10),
                "conversion": np.zeros(20, dtype=int),
            }
        )
        result = learnability(frame.iloc[:10], frame.iloc[10:], ["f0"], "conversion")
        dummy = result[result["model"] == "dummy"]
        fitted = result[result["model"] != "dummy"]
        self.assertTrue(dummy["roc_auc"].isna().all())
        self.assertTrue(dummy["average_precision"].isna().all())
        self.assertTrue((fitted["status"] == "skipped_single_class_train").all())


if __name__ == "__main__":
    unittest.main()
