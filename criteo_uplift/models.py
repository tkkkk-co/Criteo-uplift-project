# 定义 S、T、X 三类增量学习器，只使用处理前特征。
# 拟合只使用训练数据；评价集不参与预处理、拟合或提前停止。

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_feature_array —— 检查特征矩阵的形状与数值。
# 输入：
#     X：处理前特征矩阵，行对应样本，列对应特征。
# 输出：
#     values：保留数值精度的二维特征数组。
# 说明：保留输入精度；允许由后续插补处理的缺失值，拒绝无穷值和空矩阵。
#-----------------------------------------------------------------------------------------------------------------------
def _feature_array(X):
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix.")
    if values.dtype.kind not in "biuf":
        raise ValueError("X must contain numeric pre-treatment features only.")
    if np.isinf(values).any():
        raise ValueError("X contains infinity; missing values must be NaN.")
    return values


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_binary_vector —— 检查并转换一个二分类标签数组。
# 输入：
#     values：需要检查或处理的数值数组。
#     name：需要构造、读取或保存的对象名称。
#     n_rows：预期数据行数。
# 输出：
#     标签数组：与样本数一致的整数标签。
# 说明：检查标签数量与样本数量一致，并拒绝缺失值及非二分类取值。
#-----------------------------------------------------------------------------------------------------------------------
def _binary_vector(values, name, n_rows):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError(f"{name} must be a one-dimensional vector matching X.")
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1, with no missing values.")
    return values.astype(np.int8, copy=False)


#-----------------------------------------------------------------------------------------------------------------------
# 类：_ConstantProbabilityClassifier —— 训练结果只有一个类别时，提供定义明确的常数概率预测。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：训练结果只有一个类别时，提供定义明确的常数概率预测。
#-----------------------------------------------------------------------------------------------------------------------
class _ConstantProbabilityClassifier(ClassifierMixin, BaseEstimator):

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：fit —— 使用输入数据拟合当前模型。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     y：每行的二分类结果标签。
    # 输出：
    #     self：完成拟合的当前模型对象。
    # 说明：保存训练结果的平均值，后续对所有输入行返回同一个正例概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def fit(self, X, y):
        self.probability_ = float(np.mean(y))
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.n_features_in_ = X.shape[1]
        return self

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict_proba —— 生成两类标签的预测概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     概率矩阵：每行依次包含类别 0 和类别 1 的概率。
    # 说明：生成两类标签的预测概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict_proba(self, X):
        check_is_fitted(self, "probability_")
        probabilities = np.full(len(X), self.probability_, dtype=float)
        return np.column_stack((1.0 - probabilities, probabilities))


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_positive_probability —— 按类别标签提取正例预测概率。
# 输入：
#     model：已经构造或拟合的模型对象。
#     X：处理前特征矩阵，行对应样本，列对应特征。
# 输出：
#     正例概率：与输入行顺序一致的一维数组。
# 说明：按类别标签提取正例预测概率。
#-----------------------------------------------------------------------------------------------------------------------
def _positive_probability(model, X):
    probabilities = model.predict_proba(X)
    positive_columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_columns) == 0:
        return np.zeros(len(X), dtype=float)
    return np.asarray(probabilities[:, positive_columns[0]], dtype=float)


#-----------------------------------------------------------------------------------------------------------------------
# 类：_UpliftEstimator —— 统一增量模型的输入检查、拟合状态和预测接口。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：统一增量模型的输入检查、拟合状态和预测接口。
#-----------------------------------------------------------------------------------------------------------------------
class _UpliftEstimator(BaseEstimator):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_prepare_fit —— 检查拟合输入并记录模型基本状态。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     t：每行的处理组标签，0 为 Control，1 为 Treatment。
    #     y：每行的二分类结果标签。
    # 输出：
    #     features：校验后的特征矩阵。
    #     treatment：处理标签数组。
    #     outcome：结果标签数组。
    # 说明：检查特征和标签后记录列结构、样本量及处理组比例，清除旧的拟合成功状态。
    #-----------------------------------------------------------------------------------------------------------------------
    def _prepare_fit(self, X, t, y):
        # 重新拟合前移除旧的成功标记，避免拟合失败后仍被认为可预测。
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_prepare_predict —— 检查模型状态和预测特征。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     features：与训练特征结构一致的预测矩阵。
    # 说明：确认模型已拟合，并要求预测列数及命名列顺序与训练时一致。
    #-----------------------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_fit_binary —— 拟合二分类模型或单类别常数模型。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     y：每行的二分类结果标签。
    # 输出：
    #     model：已拟合的概率预测模型。
    # 说明：结果只有一个类别时使用常数概率模型，其余情况拟合当前配置的分类器。
    #-----------------------------------------------------------------------------------------------------------------------
    def _fit_binary(self, X, y):
        model = (
            _ConstantProbabilityClassifier()
            if len(np.unique(y)) == 1
            else self._new_classifier()
        )
        return model.fit(X, y)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict —— 预测每条样本的处理效应。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     效应数组：与输入行对应的处理效应预测。
    # 说明：分别预测 Control 和 Treatment 状态下的结果概率，再计算两者之差。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict(self, X):
        mu0, mu1 = self.predict_outcomes(X)
        return mu1 - mu0


#-----------------------------------------------------------------------------------------------------------------------
# 类：_LogisticConfiguration —— 保存 Logistic 配置并创建包含预处理的分类器。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：保存 Logistic 配置并创建包含预处理的分类器。
#-----------------------------------------------------------------------------------------------------------------------
class _LogisticConfiguration:
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：__init__ —— 保存模型或测试对象的初始化配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     random_state：传给模型的随机种子。
    #     max_iter：Logistic 回归最大迭代次数。
    #     C：Logistic 回归的正则化强度倒数。
    # 输出：
    #     返回值：None；参数保存在当前对象中。
    # 说明：保存模型或测试对象的初始化配置。
    #-----------------------------------------------------------------------------------------------------------------------
    def __init__(self, random_state=42, max_iter=500, C=1.0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_new_classifier —— 构造一个尚未拟合的分类器。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     classifier：按当前配置创建的分类模型。
    # 说明：将中位数插补、标准化和 Logistic 回归组成流程；预处理参数在拟合时学习。
    #-----------------------------------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------------------------------------
# 类：SLearnerLogistic —— 用特征和处理标签共同预测结果；处理标签以加性项进入模型，不额外添加交互项。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：用特征和处理标签共同预测结果；处理标签以加性项进入模型，不额外添加交互项。
#-----------------------------------------------------------------------------------------------------------------------
class SLearnerLogistic(_LogisticConfiguration, _UpliftEstimator):

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：fit —— 使用输入数据拟合当前模型。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     t：每行的处理组标签，0 为 Control，1 为 Treatment。
    #     y：每行的二分类结果标签。
    # 输出：
    #     self：完成拟合的当前模型对象。
    # 说明：把处理标签作为额外一列加入特征，使用全部训练记录拟合一个结果模型。
    #-----------------------------------------------------------------------------------------------------------------------
    def fit(self, X, t, y):
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self.model_ = self._fit_binary(np.column_stack((features, treatment)), outcome)
        self.is_fitted_ = True
        return self

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict_outcomes —— 预测两种处理状态下的结果概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     mu0：Control 状态下的预测结果概率。
    #     mu1：Treatment 状态下的预测结果概率。
    # 说明：对同一批特征分别设置两种处理标签，得到两种状态下的结果概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict_outcomes(self, X):
        features = self._prepare_predict(X)
        mu0 = _positive_probability(
            self.model_, np.column_stack((features, np.zeros(len(features))))
        )
        mu1 = _positive_probability(
            self.model_, np.column_stack((features, np.ones(len(features))))
        )
        return mu0, mu1


#-----------------------------------------------------------------------------------------------------------------------
# 类：_TwoOutcomeModels —— 分别拟合 Control 和 Treatment 的结果模型并提供两种状态的预测。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：分别拟合 Control 和 Treatment 的结果模型并提供两种状态的预测。
#-----------------------------------------------------------------------------------------------------------------------
class _TwoOutcomeModels(_UpliftEstimator):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_fit_outcomes —— 分别拟合两组的结果预测模型。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     features：需要使用的特征名列表或特征矩阵。
    #     treatment：每行的二分类处理标签。
    #     outcome：每行的二分类结果标签数组。
    # 输出：
    #     返回值：None；两个结果模型保存在当前对象中。
    # 说明：将训练数据按处理标签分为两组，各自拟合独立的结果模型。
    #-----------------------------------------------------------------------------------------------------------------------
    def _fit_outcomes(self, features, treatment, outcome):
        control = treatment == 0
        treated = treatment == 1
        self.mu0_model_ = self._fit_binary(features[control], outcome[control])
        self.mu1_model_ = self._fit_binary(features[treated], outcome[treated])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：fit —— 使用输入数据拟合当前模型。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     t：每行的处理组标签，0 为 Control，1 为 Treatment。
    #     y：每行的二分类结果标签。
    # 输出：
    #     self：完成拟合的当前模型对象。
    # 说明：先检查输入，再分别拟合两组结果模型并记录拟合成功状态。
    #-----------------------------------------------------------------------------------------------------------------------
    def fit(self, X, t, y):
        features, treatment, outcome = self._prepare_fit(X, t, y)
        self._fit_outcomes(features, treatment, outcome)
        self.is_fitted_ = True
        return self

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict_outcomes —— 预测两种处理状态下的结果概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     mu0：Control 状态下的预测结果概率。
    #     mu1：Treatment 状态下的预测结果概率。
    # 说明：把同一批输入分别交给两组模型，得到两种处理状态的预测结果。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict_outcomes(self, X):
        features = self._prepare_predict(X)
        return (
            _positive_probability(self.mu0_model_, features),
            _positive_probability(self.mu1_model_, features),
        )


#-----------------------------------------------------------------------------------------------------------------------
# 类：TLearnerLogistic —— 在两组内分别拟合预处理和 Logistic 结果模型。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：在两组内分别拟合预处理和 Logistic 结果模型。
#-----------------------------------------------------------------------------------------------------------------------
class TLearnerLogistic(_LogisticConfiguration, _TwoOutcomeModels):
    pass


#-----------------------------------------------------------------------------------------------------------------------
# 类：_XGBoostConfiguration —— 保存提升树参数并按需创建分类器或回归器。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：保存提升树参数并按需创建分类器或回归器。
#-----------------------------------------------------------------------------------------------------------------------
class _XGBoostConfiguration:
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：__init__ —— 保存模型或测试对象的初始化配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     random_state：传给模型的随机种子。
    #     n_estimators：XGBoost 使用的树数量。
    #     max_depth：每棵树的最大深度。
    #     learning_rate：提升树的学习率。
    #     min_child_weight：子节点所需的最小权重和。
    #     subsample：每棵树使用的训练行采样比例。
    #     colsample_bytree：每棵树使用的特征采样比例。
    #     tree_method：XGBoost 的树构建方式。
    #     n_jobs：模型使用的并行线程数。
    # 输出：
    #     返回值：None；参数保存在当前对象中。
    # 说明：保存模型或测试对象的初始化配置。
    #-----------------------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_tree_parameters —— 整理提升树的共用配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     参数字典：树规模、采样比例、学习率、线程数和随机种子等设置。
    # 说明：整理提升树的共用配置。
    #-----------------------------------------------------------------------------------------------------------------------
    def _tree_parameters(self):
        return self.get_params(deep=False)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_xgboost_classes —— 按需加载 XGBoost 类。
    # 输入：
    #     无。
    # 输出：
    #     classifier：XGBoost 分类器类。
    #     regressor：XGBoost 回归器类。
    # 说明：仅在需要构造提升树时加载 XGBoost，缺少运行依赖时给出明确错误。
    #-----------------------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_new_classifier —— 构造一个尚未拟合的分类器。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     classifier：按当前配置创建的分类模型。
    # 说明：构造一个尚未拟合的分类器。
    #-----------------------------------------------------------------------------------------------------------------------
    def _new_classifier(self):
        classifier, _ = self._xgboost_classes()
        return classifier(
            **self._tree_parameters(),
            objective="binary:logistic",
            eval_metric="logloss",
        )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_new_regressor —— 构造一个尚未拟合的回归器。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     regressor：按当前配置创建的回归模型。
    # 说明：构造一个尚未拟合的回归器。
    #-----------------------------------------------------------------------------------------------------------------------
    def _new_regressor(self):
        _, regressor = self._xgboost_classes()
        return regressor(
            **self._tree_parameters(), objective="reg:squarederror", eval_metric="rmse"
        )


#-----------------------------------------------------------------------------------------------------------------------
# 类：TLearnerXGBoost —— 分别使用 XGBoost 拟合两组结果，树模型原生处理缺失特征。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：分别使用 XGBoost 拟合两组结果，树模型原生处理缺失特征。
#-----------------------------------------------------------------------------------------------------------------------
class TLearnerXGBoost(_XGBoostConfiguration, _TwoOutcomeModels):
    pass


#-----------------------------------------------------------------------------------------------------------------------
# 类：XLearnerXGBoost —— 先拟合两组结果，再用跨组预测构造效应目标，最后按训练分组比例组合两组效应模型。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存配置与拟合状态的对象。
# 说明：先拟合两组结果，再用跨组预测构造效应目标，最后按训练分组比例组合两组效应模型。
#-----------------------------------------------------------------------------------------------------------------------
class XLearnerXGBoost(_XGBoostConfiguration, _TwoOutcomeModels):

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：fit —— 使用输入数据拟合当前模型。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     t：每行的处理组标签，0 为 Control，1 为 Treatment。
    #     y：每行的二分类结果标签。
    # 输出：
    #     self：完成拟合的当前模型对象。
    # 说明：先拟合两组结果模型，再用跨组预测构造伪效应，分别训练两组效应回归模型。
    #-----------------------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict —— 预测每条样本的处理效应。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     效应数组：与输入行对应的处理效应预测。
    # 说明：按训练时的处理分配比例组合两组效应预测，保留未截断的预测值。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict(self, X):
        features = self._prepare_predict(X)
        tau0 = np.asarray(self.tau0_model_.predict(features), dtype=float)
        tau1 = np.asarray(self.tau1_model_.predict(features), dtype=float)
        e = self.treatment_rate_
        # 回归预测的效应可能超出概率差的边界，这里保留原始预测。
        # 静默截断可能改变排序并掩盖模型需要检查的现象。
        return e * tau0 + (1.0 - e) * tau1


#-----------------------------------------------------------------------------------------------------------------------
# 函数：make_model —— 按名称创建正式增量模型。
# 输入：
#     name：需要构造、读取或保存的对象名称。
#     random_state：传给模型的随机种子。
#     **kwargs：传递给被调用函数的额外命名参数。
# 输出：
#     模型对象：尚未拟合的 S、T 或 X 学习器。
# 说明：根据固定名称选择对应学习器，未知名称直接报错；此步骤不训练模型。
#-----------------------------------------------------------------------------------------------------------------------
def make_model(name, random_state=42, **kwargs):
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
