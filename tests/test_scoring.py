"""Offline scoring checks using tiny, trusted synthetic joblib models."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from criteo_uplift import data
from criteo_uplift.data import FEATURES, fingerprint
from criteo_uplift.evaluation import _rank
from score_policy import score_policy


class _FeatureOnlyModel:
    n_features_in_ = 12

    def __init__(self, constant=False, max_rows=7):
        self.constant = constant
        self.max_rows = max_rows

    def _check_features(self, X):
        if list(X.columns) != FEATURES or len(X) > self.max_rows:
            raise AssertionError(
                "Only ordered pre-treatment features in bounded chunks are allowed."
            )

    def predict(self, X):
        self._check_features(X)
        return np.ones(len(X)) if self.constant else X.f0.to_numpy()

    def predict_outcomes(self, X):
        self._check_features(X)
        # Opposite rankings let the test distinguish mu1 from uplift or mu0.
        score = X.f0.to_numpy() / 100.0
        return score * 2, score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="criteo_scoring_test_")
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.source = self.root / "users.csv"
        frame = pd.DataFrame(
            {feature: np.arange(23, dtype=float) for feature in FEATURES}
        )
        frame["treatment"] = "unused text"
        frame["exposure"] = "post-treatment value"
        frame["conversion"] = "outcome must not be used"
        frame.to_csv(self.source, index=False)
        self.output = self.root / "results" / "scores.csv"
        self._save_run("t_logistic")

    def tearDown(self):
        self.directory.cleanup()

    def _save_run(self, policy, model=None):
        selection = {"selected_policy": policy, "frozen_before_test": True}
        manifest = {"features": FEATURES, "seed": 42, "selection": selection}
        (self.run / "selection.json").write_text(json.dumps(selection))
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        if policy != "random":
            name = "t_xgboost" if policy == "propensity" else policy
            joblib.dump(model or _FeatureOnlyModel(), self.run / f"{name}.joblib")

    def _score(self, **kwargs):
        return score_policy(self.run, self.source, self.output, chunk_size=7, **kwargs)

    def test_feature_only_chunks_exact_budget_and_input_unchanged(self):
        before = fingerprint(self.source)
        result = self._score(budget=0.2)
        output = pd.read_csv(self.output)
        self.assertEqual(list(output), ["source_row_id", "score", "selected"])
        np.testing.assert_array_equal(output.source_row_id, np.arange(23))
        self.assertEqual(result["selected_n"], 4)
        self.assertEqual(int(output.selected.sum()), 4)
        np.testing.assert_array_equal(
            output.loc[output.selected == 1, "source_row_id"], [19, 20, 21, 22]
        )
        self.assertEqual(fingerprint(self.source), before)

    def test_propensity_uses_treated_response_probability(self):
        self._save_run("propensity")
        self._score(budget=0.2)
        output = pd.read_csv(self.output)
        np.testing.assert_allclose(output.score, np.arange(23) / 100.0)
        np.testing.assert_array_equal(
            output.loc[output.selected == 1, "source_row_id"], [19, 20, 21, 22]
        )

    def test_random_scores_and_tie_breaking_are_chunk_independent(self):
        for policy, model in (
            ("random", None),
            ("t_logistic", _FeatureOnlyModel(constant=True)),
        ):
            with self.subTest(policy=policy):
                self._save_run(policy, model)
                path1 = self.root / f"{policy}_1.csv"
                path2 = self.root / f"{policy}_2.csv"
                score_policy(self.run, self.source, path1, budget=0.5, chunk_size=7)
                score_policy(self.run, self.source, path2, budget=0.5, chunk_size=4)
                first, second = pd.read_csv(path1), pd.read_csv(path2)
                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(int(first.selected.sum()), 11)
                self.assertNotEqual(
                    first.loc[first.selected == 1, "source_row_id"].tolist(),
                    list(range(11)),
                )

    def test_budget_boundaries_and_decimal_floor(self):
        self._save_run("random")
        frame = pd.DataFrame({feature: np.arange(100) for feature in FEATURES})
        frame.to_csv(self.source, index=False)
        for budget, expected in ((0, 0), (1, 100), (0.58, 58)):
            path = self.root / f"budget_{budget}.csv"
            result = score_policy(
                self.run, self.source, path, budget=budget, chunk_size=7
            )
            self.assertEqual(result["selected_n"], expected)
            self.assertEqual(pd.read_csv(path).selected.sum(), expected)
        for budget in (-0.1, 1.1, np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                self._score(budget=budget)

    def test_tied_score_selection_matches_evaluation_for_same_row_order(self):
        self._save_run("t_logistic", _FeatureOnlyModel(constant=True))
        result = self._score(budget=0.2)
        output = pd.read_csv(self.output)
        expected = np.zeros(len(output), dtype=np.int8)
        expected[_rank(output.score.to_numpy(), seed=42)[: result["selected_n"]]] = 1
        np.testing.assert_array_equal(output.selected, expected)

    def test_readonly_input_data_symlink_and_existing_output_guards(self):
        before = fingerprint(self.source)
        with self.assertRaises(ValueError):
            score_policy(self.run, self.source, self.source)
        protected = self.root / "protected_data"
        protected.mkdir()
        link = self.root / "alias"
        link.symlink_to(protected, target_is_directory=True)
        with patch.object(data, "DATA_ROOT", protected):
            for path in (protected / "out.csv", link / "out.csv"):
                with self.assertRaises(ValueError):
                    score_policy(self.run, self.source, path)
        self.output.parent.mkdir()
        self.output.write_text("existing output")
        with self.assertRaises(FileExistsError):
            self._score()
        self.assertEqual(self.output.read_text(), "existing output")
        self.assertEqual(fingerprint(self.source), before)
        self.assertFalse((protected / "out.csv").exists())

    def test_missing_run_model_features_and_unfrozen_selection_have_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "run"):
            score_policy(self.root / "missing", self.source, self.output)
        (self.run / "t_logistic.joblib").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "模型"):
            self._score()
        self._save_run("random")
        frame = pd.read_csv(self.source).drop(columns="f11")
        frame.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "f11"):
            self._score()
        frame["f11"] = 0
        frame.to_csv(self.source, index=False)
        (self.run / "selection.json").write_text(
            json.dumps({"selected_policy": "random"})
        )
        with self.assertRaisesRegex(ValueError, "frozen_before_test"):
            self._score()
        (self.run / "selection.json").write_text(
            json.dumps({"selected_policy": "random", "frozen_before_test": False})
        )
        with self.assertRaisesRegex(ValueError, "frozen_before_test"):
            self._score()
        self.assertFalse(self.output.exists())

    def test_invalid_features_and_excessive_chunk_size_fail_before_output(self):
        self._save_run("random")
        frame = pd.read_csv(self.source)
        frame.loc[10, "f0"] = np.nan
        frame.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "前置特征"):
            self._score()
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            score_policy(self.run, self.source, self.output, chunk_size=100_001)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
