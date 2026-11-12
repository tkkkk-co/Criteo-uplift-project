#-------------
# S/T/X 算法与结果分类器、效应回归器分离；所有预处理只在训练数据 fit。
#-------------
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


def make_classifier_model(name, **params):
    #-------------
    # 函数名：make_classifier_model
    # 功能：创建以 0/1 为训练标签、可输出结果概率的分类模型及其预处理。
    # 输入参数：
    #     - name：分类器名称，当前支持 logistic 和 xgboost。
    #     - **params：传递给对应分类器构造函数的参数。
    # 输出参数：
    #     - 未拟合的结果分类器或 Pipeline。
    #-------------
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
    raise ValueError(f"不支持的结果模型：{name}")


def make_regression_model(name, **params):
    #-------------
    # 函数名：make_regression_model
    # 功能：创建用于拟合 X-Learner 连续伪效应标签的回归模型及其预处理。
    # 输入参数：
    #     - name：回归器名称，linear 为线性回归，xgboost 为提升树回归。
    #     - **params：传递给对应回归器构造函数的参数。
    # 输出参数：
    #     - 未拟合的效应回归器或 Pipeline。
    #-------------
    if name == "linear":
        params.pop("random_state", None)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()), ("regressor", LinearRegression(**params))])
    if name == "xgboost":
        configure_runtime()
        from xgboost import XGBRegressor
        return XGBRegressor(**{**_XGB_DEFAULTS, "objective": "reg:squarederror", "eval_metric": "rmse", **params})
    raise ValueError(f"不支持的效应模型：{name}")

class _ConstantProbabilityClassifier(ClassifierMixin, BaseEstimator):

    def fit(self, X, y):
        #-------------
        # 函数名：_ConstantProbabilityClassifier.fit
        # 功能：将训练样本的正例比例保存为常数概率模型，并记录输入特征数量。
        # 输入参数：
        #     - X：训练特征矩阵。
        #     - y：训练二元标签。
        # 输出参数：
        #     - 拟合后的分类器自身。
        #-------------
        self.probability_ = float(np.mean(y))
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        #-------------
        # 函数名：_ConstantProbabilityClassifier.predict_proba
        # 功能：对每一行返回常数模型的负类和正类概率。
        # 输入参数：
        #     - X：需要预测概率的特征矩阵。
        # 输出参数：
        #     - 形状为 n_samples × 2 的概率矩阵，列顺序对应 classes_。
        #-------------
        check_is_fitted(self, "probability_")
        probabilities = np.full(len(X), self.probability_, dtype=float)
        return np.column_stack((1.0 - probabilities, probabilities))

def _positive_probability(model, X):
    #-------------
    # 函数名：_positive_probability
    # 功能：从分类器概率矩阵中按 classes_ 标签定位并提取正类 1 的概率。
    # 输入参数：
    #     - model：已拟合且提供 predict_proba/classes_ 的分类器。
    #     - X：需要计算正类概率的特征矩阵。
    # 输出参数：
    #     - 每行属于正类 1 的概率向量；若模型没有正类列则返回全零向量。
    #-------------
    probabilities = model.predict_proba(X)
    positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_columns) == 0:
        return np.zeros(len(X), dtype=float)
    return np.asarray(probabilities[:, positive_columns[0]], dtype=float)

class _BaseUpliftLearner(BaseEstimator):
    def __init__(self, outcome_model="logistic", effect_model=None,
                 outcome_params=None, effect_params=None):
        #-------------
        # 函数名：_BaseUpliftLearner.__init__
        # 功能：保存结果模型、效应模型及两类基模型各自的构造参数。
        # 输入参数：
        #     - outcome_model：用于估计潜在结果的分类模型名称。
        #     - effect_model：用于估计伪处理效应的回归模型名称；S/T 可不设置。
        #     - outcome_params：结果分类器的参数字典。
        #     - effect_params：效应回归器的参数字典。
        # 输出参数：
        #     - 无；初始化 Learner 实例属性。
        #-------------
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.outcome_params = outcome_params
        self.effect_params = effect_params

    def _new_classifier(self):
        #-------------
        # 函数名：_BaseUpliftLearner._new_classifier
        # 功能：根据当前 Learner 的结果模型配置创建一个新的结果分类器。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 未拟合的 Outcome Model 分类器。
        #-------------
        return make_classifier_model(self.outcome_model, **(self.outcome_params or {}))

    def _new_regressor(self):
        #-------------
        # 函数名：_BaseUpliftLearner._new_regressor
        # 功能：根据当前 Learner 的效应模型配置创建一个新的效应回归器。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 未拟合的 Effect Model 回归器。
        #-------------
        return make_regression_model(self.effect_model, **(self.effect_params or {}))

    def _prepare_fit(self, X, t, y):
        #-------------
        # 函数名：_BaseUpliftLearner._prepare_fit
        # 功能：校验拟合输入并转换 X、T、Y，同时记录特征数、样本数和训练处理比例。
        # 输入参数：
        #     - X：训练特征矩阵，可包含后续由 Pipeline 插补的 NaN。
        #     - t：训练样本的 0/1 处理分配向量。
        #     - y：训练样本的 0/1 结果变量向量。
        # 输出参数：
        #     - 三元组：NumPy 特征矩阵、int8 处理分配数组和 int8 结果数组。
        #-------------
        # 重新拟合前移除旧的成功标记，避免拟合失败后仍被认为可预测。
        self.__dict__.pop("is_fitted_", None)
        check_feature_matrix(X, allow_nan=True)
        features = np.asarray(X)
        check_binary_vector(t, "t", len(features))
        check_binary_vector(y, "y", len(features))
        treatment = np.asarray(t, dtype=np.int8)
        outcome = np.asarray(y, dtype=np.int8)
        if len(np.unique(treatment)) != 2:
            raise ValueError("训练数据必须同时包含处理组和对照组。")
        self.n_features_in_ = features.shape[1]
        self.n_samples_ = len(features)
        self.treatment_rate_ = float(treatment.mean())
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.__dict__.pop("feature_names_in_", None)
        return features, treatment, outcome

    def _prepare_predict(self, X):
        #-------------
        # 函数名：_BaseUpliftLearner._prepare_predict
        # 功能：检查模型已拟合、预测特征合法，并验证特征数量和列顺序与训练阶段一致。
        # 输入参数：
        #     - X：待预测的特征矩阵或 DataFrame。
        # 输出参数：
        #     - 通过校验后的 NumPy 特征矩阵。
        #-------------
        check_is_fitted(self, "is_fitted_")
        check_feature_matrix(X, allow_nan=True)
        features = np.asarray(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(f"X 必须包含 {self.n_features_in_} 个特征。")
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError(
                    "预测列及其顺序必须与训练时一致。"
                )
        return features

    def _fit_binary(self, X, y):
        #-------------
        # 函数名：_BaseUpliftLearner._fit_binary
        # 功能：根据训练标签是否只有一个类别，选择常数概率分类器或正常 Outcome Model 并完成拟合。
        # 输入参数：
        #     - X：当前模型训练使用的特征矩阵。
        #     - y：当前模型训练使用的二元结果标签。
        # 输出参数：
        #     - 已经拟合的分类器实例。
        #-------------
        model = (
            _ConstantProbabilityClassifier()
            if len(np.unique(y)) == 1
            else self._new_classifier()
        )
        return model.fit(X, y)

    def predict(self, X):
        #-------------
        # 函数名：_BaseUpliftLearner.predict
        # 功能：分别预测 T=0 和 T=1 下的潜在结果概率，并以两者之差作为 CATE。
        # 输入参数：
        #     - X：需要估计个体处理效应的特征矩阵。
        # 输出参数：
        #     - 每个样本的 CATE 预测向量，即 mu1 - mu0。
        #-------------
        mu0, mu1 = self.predict_outcomes(X)
        return mu1 - mu0

class SLearner(_BaseUpliftLearner):

    def fit(self, X, t, y):
        #-------------
        # 函数名：SLearner.fit
        # 功能：将 X 与 T 拼接后拟合一个统一的结果模型，完成 S-Learner 训练。
        # 输入参数：
        #     - X：训练特征矩阵。
        #     - t：训练处理分配向量。
        #     - y：训练二元结果变量。
        # 输出参数：
        #     - 拟合完成的 S-Learner 实例。
        #-------------
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self.model_ = self._fit_binary(np.column_stack((features, treatment)), outcome)
        self.is_fitted_ = True
        return self

    def predict_outcomes(self, X):
        #-------------
        # 函数名：SLearner.predict_outcomes
        # 功能：对同一批 X 分别将 T 设为 0 和 1，预测两种潜在结果概率。
        # 输入参数：
        #     - X：需要预测潜在结果的特征矩阵。
        # 输出参数：
        #     - 二元组：(control 潜在结果概率 mu0, treatment 潜在结果概率 mu1)。
        #-------------
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
        #-------------
        # 函数名：TLearner._fit_outcomes
        # 功能：分别使用 Control 和 Treatment 子样本拟合两个独立的结果分类器。
        # 输入参数：
        #     - features：已校验的训练特征矩阵。
        #     - treatment：0/1 处理分配数组。
        #     - outcome：0/1 结果数组。
        # 输出参数：
        #     - 无；将两个已拟合模型保存为实例属性 mu0_model_ 和 mu1_model_。
        #-------------
        control = treatment == 0
        treated = treatment == 1
        self.mu0_model_ = self._fit_binary(features[control], outcome[control])
        self.mu1_model_ = self._fit_binary(features[treated], outcome[treated])

    def fit(self, X, t, y):
        #-------------
        # 函数名：TLearner.fit
        # 功能：分别在 Control 和 Treatment 样本上拟合两个结果模型，完成 T-Learner 训练。
        # 输入参数：
        #     - X：训练特征矩阵。
        #     - t：训练处理分配向量。
        #     - y：训练二元结果变量。
        # 输出参数：
        #     - 拟合完成的 T-Learner 实例。
        #-------------
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self._fit_outcomes(features, treatment, outcome)
        self.is_fitted_ = True
        return self

    def predict_outcomes(self, X):
        #-------------
        # 函数名：TLearner.predict_outcomes
        # 功能：调用 Control 与 Treatment 两个结果模型，返回两种潜在结果概率。
        # 输入参数：
        #     - X：需要预测潜在结果的特征矩阵。
        # 输出参数：
        #     - 二元组：(control 潜在结果概率 mu0, treatment 潜在结果概率 mu1)。
        #-------------
        features = self._prepare_predict(X)
        return (
            _positive_probability(self.mu0_model_, features),
            _positive_probability(self.mu1_model_, features),
        )

class XLearner(TLearner):

    def fit(self, X, t, y):
        #-------------
        # 函数名：XLearner.fit
        # 功能：拟合两组 Outcome Model，构造跨组伪处理效应，并训练两个 Effect Model。
        # 输入参数：
        #     - X：训练特征矩阵。
        #     - t：训练处理分配向量。
        #     - y：训练二元结果变量。
        # 输出参数：
        #     - 拟合完成的 X-Learner 实例，包含两组结果模型和两组效应回归器。
        #-------------
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
        #-------------
        # 函数名：XLearner.predict
        # 功能：使用两组效应回归器预测伪处理效应，并按训练集 Treatment 比例加权得到 CATE。
        # 输入参数：
        #     - X：需要估计个体处理效应的特征矩阵。
        # 输出参数：
        #     - 每个样本的加权 CATE 预测向量。
        #-------------
        features = self._prepare_predict(X)
        tau0 = np.asarray(self.tau0_model_.predict(features), dtype=float)
        tau1 = np.asarray(self.tau1_model_.predict(features), dtype=float)
        e = self.treatment_rate_
        # 回归预测的效应可能超出概率差的边界，这里保留原始预测。
        # 静默截断可能改变排序并掩盖模型需要检查的现象。
        return e * tau0 + (1.0 - e) * tau1


def make_uplift_model(learner, outcome_model, effect_model=None, *,
                      outcome_params=None, effect_params=None, **params):
    #-------------
    # 函数名：make_uplift_model
    # 功能：根据 S/T/X Learner 类型及 Outcome/Effect Base Model 配置创建可组合的 Uplift Learner。
    # 输入参数：
    #     - learner：Meta Learner 类型，支持 s、t、x。
    #     - outcome_model：潜在结果分类器名称。
    #     - effect_model：X-Learner 使用的效应回归器名称。
    #     - outcome_params：Outcome Model 专属参数字典。
    #     - effect_params：Effect Model 专属参数字典。
    #     - **params：传递给基模型的公共参数；可由专属参数字典覆盖。
    # 输出参数：
    #     - 根据配置创建的 S、T 或 X Learner 实例。
    #-------------
    learners = {"s": SLearner, "t": TLearner, "x": XLearner}
    if learner not in learners:
        raise ValueError(f"不支持的学习器：{learner}")
    if outcome_model not in {"logistic", "xgboost"}:
        raise ValueError(f"不支持的结果模型：{outcome_model}")
    if learner == "x" and effect_model not in {"linear", "xgboost"}:
        raise ValueError("X-Learner 需要指定效应模型：linear 或 xgboost")
    if learner != "x" and (effect_model is not None or effect_params):
        raise ValueError("S/T 学习器不使用效应模型")
    return learners[learner](outcome_model=outcome_model, effect_model=effect_model,
                             outcome_params={**params, **(outcome_params or {})},
                             effect_params={**params, **(effect_params or {})} if learner == "x" else None)

def predict_policy_scores(models, X):
    #-------------
    # 函数名：predict_policy_scores
    # 功能：对模型字典中的增量模型生成 CATE 分数，并生成处理状态下结果概率的 propensity 基线分数。
    # 输入参数：
    #     - models：模型名称到已拟合 Uplift Learner 的映射。
    #     - X：需要生成策略分数的特征矩阵。
    # 输出参数：
    #     - 策略分数字典：各 Uplift 模型的 CATE，以及 propensity 响应概率分数。
    #-------------
    scores = {name: model.predict(X) for name, model in models.items()}
    # propensity 是 P(Y=1|X,T=1)，不是处理分配概率 P(T=1|X)。
    scores["propensity"] = models["t_xgboost"].predict_outcomes(X)[1]
    return scores
