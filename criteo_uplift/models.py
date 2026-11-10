"""S/T/X 算法与结果分类器、效应回归器分离；所有预处理只在训练数据 fit。"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from .input_checks import check_binary_vector, check_feature_matrix
from .config import configure_runtime

_XGB_DEFAULTS = dict(random_state=42, n_estimators=150, max_depth=4, learning_rate=.05,
                     min_child_weight=20, subsample=.8, colsample_bytree=.9,
                     tree_method="hist", n_jobs=4)


def make_outcome_model(name, **params):
    """创建 Y 的分类器，默认参数与原正式实验一致。"""
    if name == "logistic":
        settings = dict(random_state=42, max_iter=500, C=1., solver="lbfgs")
        settings.update(params)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()),
                         ("classifier", LogisticRegression(**settings))])
    if name == "xgboost":
        configure_runtime()
        from xgboost import XGBClassifier
        settings = {**_XGB_DEFAULTS, "objective": "binary:logistic", "eval_metric": "logloss", **params}
        return XGBClassifier(**settings)
    raise ValueError(f"Unknown outcome model: {name}")


def make_effect_model(name, **params):
    """创建连续伪效应的回归器；Linear 是新增可组合能力，不进入正式候选。"""
    if name == "linear":
        params.pop("random_state", None)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()), ("regressor", LinearRegression(**params))])
    if name == "xgboost":
        configure_runtime()
        from xgboost import XGBRegressor
        return XGBRegressor(**{**_XGB_DEFAULTS, "objective": "reg:squarederror", "eval_metric": "rmse", **params})
    raise ValueError(f"Unknown effect model: {name}")

class _ConstantProbabilityClassifier(ClassifierMixin, BaseEstimator):

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
    probabilities = model.predict_proba(X)
    positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_columns) == 0:
        return np.zeros(len(X), dtype=float)
    return np.asarray(probabilities[:, positive_columns[0]], dtype=float)

class _BaseUpliftLearner(BaseEstimator):
    def __init__(self, outcome_model="logistic", effect_model=None,
                 outcome_params=None, effect_params=None):
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.outcome_params = outcome_params
        self.effect_params = effect_params

    def _new_classifier(self):
        return make_outcome_model(self.outcome_model, **(self.outcome_params or {}))

    def _new_regressor(self):
        return make_effect_model(self.effect_model, **(self.effect_params or {}))

    def _prepare_fit(self, X, t, y):
        # 重新拟合前移除旧的成功标记，避免拟合失败后仍被认为可预测。
        self.__dict__.pop("is_fitted_", None)
        check_feature_matrix(X, allow_nan=True)
        features = np.asarray(X)
        check_binary_vector(t, "t", len(features))
        check_binary_vector(y, "y", len(features))
        treatment = np.asarray(t, dtype=np.int8)
        outcome = np.asarray(y, dtype=np.int8)
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
        check_feature_matrix(X, allow_nan=True)
        features = np.asarray(X)
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

class SLearner(_BaseUpliftLearner):

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

class TLearner(_BaseUpliftLearner):
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

class XLearner(TLearner):

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
        # 回归预测的效应可能超出概率差的边界，这里保留原始预测。
        # 静默截断可能改变排序并掩盖模型需要检查的现象。
        return e * tau0 + (1.0 - e) * tau1


def make_uplift_model(learner, outcome_model, effect_model=None, *,
                      outcome_params=None, effect_params=None, **params):
    """组合算法与基模型；公共参数可用专属参数字典分别覆盖。"""
    learners = {"s": SLearner, "t": TLearner, "x": XLearner}
    if learner not in learners:
        raise ValueError(f"Unknown learner: {learner}")
    if outcome_model not in {"logistic", "xgboost"}:
        raise ValueError(f"Unknown outcome model: {outcome_model}")
    if learner == "x" and effect_model not in {"linear", "xgboost"}:
        raise ValueError("X-Learner requires an effect model: linear or xgboost")
    if learner != "x" and (effect_model is not None or effect_params):
        raise ValueError("S/T learners do not use an effect model")
    return learners[learner](outcome_model=outcome_model, effect_model=effect_model,
                             outcome_params={**params, **(outcome_params or {})},
                             effect_params={**params, **(effect_params or {})} if learner == "x" else None)

def predict_policy_scores(models, X):
    """生成各增量模型的分数，复用 T-XGBoost 的处理组模型作为响应概率基线。"""
    scores = {name: model.predict(X) for name, model in models.items()}
    # propensity 是 P(Y=1|X,T=1)，不是处理分配概率 P(T=1|X)。
    scores["propensity"] = models["t_xgboost"].predict_outcomes(X)[1]
    return scores
