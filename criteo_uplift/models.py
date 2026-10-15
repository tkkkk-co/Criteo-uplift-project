"""Small, auditable uplift estimators for randomized binary treatments.

All learners use ``fit(X, t, y)``, ``predict(X)`` for CATE, and
``predict_outcomes(X)`` for ``(mu0, mu1)``. Only pre-treatment features
belong in X. They do not reweight outcomes, resample classes, tune on test
data, or use test-set early stopping. XGBoost is imported only when needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _feature_array(X):
    """Preserve numeric precision; allow NaN but reject infinities."""
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix.")
    if values.dtype.kind not in "biuf":
        raise ValueError("X must contain numeric pre-treatment features only.")
    if np.isinf(values).any():
        raise ValueError("X contains infinity; missing values must be NaN.")
    return values


def _binary_vector(values, name, n_rows):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{name} must be a one-dimensional vector matching X.")
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1, with no missing values.")
    return values.astype(np.int8, copy=False)


class _ConstantProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """A defined probability model when a training arm has one outcome class."""

    def fit(self, X, y):
        self.probability_ = float(np.mean(y))
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "probability_")
        probabilities = np.full(len(X), self.probability_, dtype=float)
        return np.column_stack((1.0 - probabilities, probabilities))


def _positive_probability(model, X):
    """Select label 1 explicitly instead of assuming a probability-column order."""
    probabilities = model.predict_proba(X)
    positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_columns) == 0:
        return np.zeros(len(X), dtype=float)
    return np.asarray(probabilities[:, positive_columns[0]], dtype=float)


class _UpliftEstimator(BaseEstimator):
    def _prepare_fit(self, X, t, y):
        # Remove a previous success flag before a refit that may fail.
        self.__dict__.pop("is_fitted_", None)
        features = _feature_array(X)
        treatment = _binary_vector(t, "t", len(features))
        outcome = _binary_vector(y, "y", len(features))
        if len(np.unique(treatment)) != 2:
            raise ValueError("Training data must contain both treatment arms.")
        self.n_features_in_ = features.shape[1]
        self.n_samples_ = len(features)
        self.treatment_rate_ = float(treatment.mean())
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.__dict__.pop("feature_names_in_", None)
        return features, treatment, outcome

    def _prepare_predict(self, X):
        check_is_fitted(self, "is_fitted_")
        features = _feature_array(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features.")
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError(
                    "Prediction columns must match training columns in order."
                )
        return features

    def _fit_binary(self, X, y):
        model = (
            _ConstantProbabilityClassifier()
            if len(np.unique(y)) == 1
            else self._new_classifier()
        )
        return model.fit(X, y)

    def predict(self, X):
        mu0, mu1 = self.predict_outcomes(X)
        return mu1 - mu0


class _LogisticConfiguration:
    def __init__(self, random_state=42, max_iter=500, C=1.0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C

    def _new_classifier(self):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=self.random_state,
                        max_iter=self.max_iter,
                        C=self.C,
                        solver="lbfgs",
                    ),
                ),
            ]
        )


class SLearnerLogistic(_LogisticConfiguration, _UpliftEstimator):
    """Logistic S-learner with an additive treatment term and no interactions.

    Its log-odds treatment effect is constant. Probability-scale effects can
    still vary with baseline risk, but this is intentionally a limited baseline.
    """

    def fit(self, X, t, y):
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self.model_ = self._fit_binary(np.column_stack((features, treatment)), outcome)
        self.is_fitted_ = True
        return self

    def predict_outcomes(self, X):
        features = self._prepare_predict(X)
        mu0 = _positive_probability(
            self.model_, np.column_stack((features, np.zeros(len(features))))
        )
        mu1 = _positive_probability(
            self.model_, np.column_stack((features, np.ones(len(features))))
        )
        return mu0, mu1


class _TwoOutcomeModels(_UpliftEstimator):
    def _fit_outcomes(self, features, treatment, outcome):
        control = treatment == 0
        treated = treatment == 1
        self.mu0_model_ = self._fit_binary(features[control], outcome[control])
        self.mu1_model_ = self._fit_binary(features[treated], outcome[treated])

    def fit(self, X, t, y):
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self._fit_outcomes(features, treatment, outcome)
        self.is_fitted_ = True
        return self

    def predict_outcomes(self, X):
        features = self._prepare_predict(X)
        return (
            _positive_probability(self.mu0_model_, features),
            _positive_probability(self.mu1_model_, features),
        )


class TLearnerLogistic(_LogisticConfiguration, _TwoOutcomeModels):
    """Separate logistic outcome models with preprocessing fit within each arm."""


class _XGBoostConfiguration:
    def __init__(
        self,
        random_state=42,
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=4,
    ):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.n_jobs = n_jobs

    def _tree_parameters(self):
        return self.get_params(deep=False)

    @staticmethod
    def _xgboost_classes():
        from .runtime import configure_runtime

        configure_runtime()
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except (ImportError, OSError) as exc:
            raise ImportError(
                "XGBoost is unavailable. Install xgboost and its OpenMP runtime "
                "(libomp on macOS), or use the logistic baselines."
            ) from exc
        return XGBClassifier, XGBRegressor

    def _new_classifier(self):
        classifier, _ = self._xgboost_classes()
        return classifier(
            **self._tree_parameters(),
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def _new_regressor(self):
        _, regressor = self._xgboost_classes()
        return regressor(
            **self._tree_parameters(), objective="reg:squarederror", eval_metric="rmse"
        )


class TLearnerXGBoost(_XGBoostConfiguration, _TwoOutcomeModels):
    """Separate XGBoost outcome models; trees handle missing features natively."""


class XLearnerXGBoost(_XGBoostConfiguration, _TwoOutcomeModels):
    """X-learner with cross-arm imputation and randomized treatment weighting.

    d0 = mu1(X0) - y0; d1 = y1 - mu0(X1).
    tau = e * tau0 + (1 - e) * tau1, where e is the training treatment rate.

    Each counterfactual prediction uses the opposite treatment arm's model.
    ``predict_outcomes`` exposes the first-stage outcome predictions;
    ``predict`` uses the second-stage effect models and need not equal mu1-mu0.
    """

    def fit(self, X, t, y):
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self._fit_outcomes(features, treatment, outcome)
        control = treatment == 0
        treated = treatment == 1
        d0 = (
            _positive_probability(self.mu1_model_, features[control]) - outcome[control]
        )
        d1 = outcome[treated] - _positive_probability(
            self.mu0_model_, features[treated]
        )
        self.tau0_model_ = self._new_regressor().fit(features[control], d0)
        self.tau1_model_ = self._new_regressor().fit(features[treated], d1)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        features = self._prepare_predict(X)
        tau0 = np.asarray(self.tau0_model_.predict(features), dtype=float)
        tau1 = np.asarray(self.tau1_model_.predict(features), dtype=float)
        e = self.treatment_rate_
        # Do not clip effects: regression may exceed probability bounds and
        # silent clipping can alter ranking and hide a model diagnostic.
        return e * tau0 + (1.0 - e) * tau1


def make_model(name, random_state=42, **kwargs):
    """Construct one of s_logistic, t_logistic, t_xgboost, or x_xgboost."""
    estimators = {
        "s_logistic": SLearnerLogistic,
        "t_logistic": TLearnerLogistic,
        "t_xgboost": TLearnerXGBoost,
        "x_xgboost": XLearnerXGBoost,
    }
    if name not in estimators:
        raise ValueError(
            f"Unknown model {name!r}; choose from {', '.join(estimators)}."
        )
    return estimators[name](random_state=random_state, **kwargs)


__all__ = [
    "SLearnerLogistic",
    "TLearnerLogistic",
    "TLearnerXGBoost",
    "XLearnerXGBoost",
    "make_model",
]
