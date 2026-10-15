"""Fast synthetic checks for uplift-model contracts and X-learner arithmetic."""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from criteo_uplift.models import (
    SLearnerLogistic,
    TLearnerLogistic,
    TLearnerXGBoost,
    XLearnerXGBoost,
    _positive_probability,
    make_model,
)


class _FixedProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability):
        self.probability = probability

    def fit(self, X, y):
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    def predict_proba(self, X):
        return np.tile([1.0 - self.probability, self.probability], (len(X), 1))


class _RecordingRegressor:
    def __init__(self, prediction):
        self.prediction = prediction

    def fit(self, X, y):
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.prediction)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(18)
        self.X = rng.normal(size=(240, 3))
        self.t = np.tile([0, 1], 120)
        probability = 1.0 / (1.0 + np.exp(-(-0.8 + self.X[:, 0] + 0.7 * self.t)))
        self.y = rng.binomial(1, probability)

    def test_logistic_contracts_pickle_and_clone(self):
        for estimator in (SLearnerLogistic(), TLearnerLogistic()):
            with self.subTest(estimator=type(estimator).__name__):
                self.assertIs(estimator.fit(self.X, self.t, self.y), estimator)
                mu0, mu1 = estimator.predict_outcomes(self.X[:11])
                self.assertEqual(mu0.shape, (11,))
                self.assertTrue(np.all((mu0 >= 0) & (mu0 <= 1)))
                self.assertTrue(np.all((mu1 >= 0) & (mu1 <= 1)))
                np.testing.assert_allclose(estimator.predict(self.X[:11]), mu1 - mu0)
                restored = pickle.loads(pickle.dumps(estimator))
                np.testing.assert_allclose(restored.predict(self.X[:11]), mu1 - mu0)
                self.assertEqual(clone(estimator).get_params(), estimator.get_params())

    def test_s_learner_has_only_additive_treatment_feature(self):
        model = SLearnerLogistic().fit(self.X, self.t, self.y)
        self.assertEqual(
            model.model_["classifier"].coef_.shape, (1, self.X.shape[1] + 1)
        )

    def test_preprocessing_is_fit_per_training_arm_only(self):
        X = np.array([[0.0], [np.nan], [2.0], [100.0], [np.nan], [104.0]])
        t = np.array([0, 0, 0, 1, 1, 1])
        y = np.array([0, 1, 0, 1, 0, 1])
        model = TLearnerLogistic().fit(X, t, y)
        np.testing.assert_allclose(model.mu0_model_["imputer"].statistics_, [1.0])
        np.testing.assert_allclose(model.mu1_model_["imputer"].statistics_, [102.0])
        control_mean = model.mu0_model_["scaler"].mean_.copy()
        model.predict(np.array([[1e6], [np.nan]]))
        np.testing.assert_array_equal(model.mu0_model_["scaler"].mean_, control_mean)

    def test_constant_outcome_arms_have_defined_positive_probabilities(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        for estimator in (TLearnerLogistic(), TLearnerXGBoost()):
            estimator.fit(X, t, t)
            mu0, mu1 = estimator.predict_outcomes(X[:2])
            np.testing.assert_array_equal(mu0, [0, 0])
            np.testing.assert_array_equal(mu1, [1, 1])
            np.testing.assert_array_equal(estimator.predict(X[:2]), [1, 1])
        for outcome in (np.zeros(8), np.ones(8)):
            model = SLearnerLogistic().fit(X, t, outcome)
            np.testing.assert_array_equal(model.predict(X[:2]), [0, 0])

    def test_probability_selection_uses_class_label(self):
        class ReversedClasses:
            classes_ = np.array([1, 0])

            def predict_proba(self, X):
                return np.tile([0.8, 0.2], (len(X), 1))

        np.testing.assert_allclose(
            _positive_probability(ReversedClasses(), self.X[:2]), 0.8
        )

    def test_x_learner_cross_arm_targets_and_weight_direction(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        t = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        mu0, mu1 = _FixedProbabilityModel(0.2), _FixedProbabilityModel(0.8)
        tau0, tau1 = _RecordingRegressor(0.3), _RecordingRegressor(0.7)
        model = XLearnerXGBoost()
        with patch.object(
            model, "_new_classifier", side_effect=[mu0, mu1]
        ), patch.object(model, "_new_regressor", side_effect=[tau0, tau1]):
            model.fit(X, t, y)
        np.testing.assert_array_equal(mu0.fit_X, X[t == 0])
        np.testing.assert_array_equal(mu1.fit_X, X[t == 1])
        np.testing.assert_allclose(tau0.fit_y, 0.8 - y[t == 0])
        np.testing.assert_allclose(tau1.fit_y, y[t == 1] - 0.2)
        self.assertEqual(model.treatment_rate_, 0.25)
        np.testing.assert_allclose(model.predict(X[:2]), 0.25 * 0.3 + 0.75 * 0.7)
        np.testing.assert_allclose(
            model.predict_outcomes(X[:2]), [[0.2, 0.2], [0.8, 0.8]]
        )

    def test_validation_and_unfitted_behavior(self):
        with self.assertRaises(NotFittedError):
            TLearnerLogistic().predict(self.X)
        invalid_cases = [
            (self.X, np.zeros(len(self.t)), self.y),
            (self.X, self.t, np.full(len(self.y), 2)),
            (self.X, self.t[:-1], self.y),
            (self.X, self.t, self.y.reshape(-1, 1)),
            (np.full(self.X.shape, np.inf), self.t, self.y),
        ]
        for X, t, y in invalid_cases:
            with self.assertRaises(ValueError):
                TLearnerLogistic().fit(X, t, y)
        model = TLearnerLogistic().fit(self.X, self.t, self.y)
        with self.assertRaises(ValueError):
            model.predict(self.X[:, :2])

    def test_factory_and_tree_configuration_need_no_xgboost_import(self):
        for name, estimator_type in (
            ("s_logistic", SLearnerLogistic),
            ("t_logistic", TLearnerLogistic),
            ("t_xgboost", TLearnerXGBoost),
            ("x_xgboost", XLearnerXGBoost),
        ):
            model = make_model(name, random_state=31)
            self.assertIsInstance(model, estimator_type)
            self.assertEqual(model.random_state, 31)
            self.assertEqual(clone(model).get_params(), model.get_params())
        with self.assertRaises(ValueError):
            make_model("unknown")


if __name__ == "__main__":
    unittest.main()
