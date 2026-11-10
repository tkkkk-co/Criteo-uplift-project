# 项目测试：验证模型接口及拟合逻辑。

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from criteo_uplift.models import SLearner, TLearner, XLearner, _positive_probability, make_uplift_model



#-----------------------------------------------------------------------------------------------------------------------
# 类：_FixedProbabilityModel —— 记录拟合输入并返回固定概率的测试模型。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：记录拟合输入并返回固定概率的测试模型。
#-----------------------------------------------------------------------------------------------------------------------
class _FixedProbabilityModel:
    classes_ = np.array([0, 1])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：__init__ —— 保存模型或测试对象的初始化配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     probability：测试或常数模型使用的正例概率。
    # 输出：
    #     返回值：None；参数保存在当前对象中。
    # 说明：保存模型或测试对象的初始化配置。
    #-----------------------------------------------------------------------------------------------------------------------
    def __init__(self, probability):
        self.probability = probability

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：fit —— 记录传入的拟合数据。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     y：每行的二分类结果标签。
    # 输出：
    #     self：记录好输入数据的当前测试模型。
    # 说明：记录传入的拟合数据。
    #-----------------------------------------------------------------------------------------------------------------------
    def fit(self, X, y):
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
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
        return np.tile([1.0 - self.probability, self.probability], (len(X), 1))


#-----------------------------------------------------------------------------------------------------------------------
# 类：_RecordingRegressor —— 记录拟合输入并返回固定数值的测试回归器。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：记录拟合输入并返回固定数值的测试回归器。
#-----------------------------------------------------------------------------------------------------------------------
class _RecordingRegressor:
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：__init__ —— 保存模型或测试对象的初始化配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     prediction：与样本对应的预测概率或测试预设分数。
    # 输出：
    #     返回值：None；参数保存在当前对象中。
    # 说明：保存模型或测试对象的初始化配置。
    #-----------------------------------------------------------------------------------------------------------------------
    def __init__(self, prediction):
        self.prediction = prediction

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：fit —— 记录传入的拟合数据。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    #     y：每行的二分类结果标签。
    # 输出：
    #     self：记录好输入数据的当前测试模型。
    # 说明：记录传入的拟合数据。
    #-----------------------------------------------------------------------------------------------------------------------
    def fit(self, X, y):
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict —— 返回测试预设的常数预测。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     预测数组：每条样本对应预设的固定数值。
    # 说明：返回测试预设的常数预测。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict(self, X):
        return np.full(len(X), self.prediction)


#-----------------------------------------------------------------------------------------------------------------------
# 类：ModelTests —— 验证模型接口及拟合逻辑。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证模型接口及拟合逻辑。
#-----------------------------------------------------------------------------------------------------------------------
class ModelTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
        rng = np.random.default_rng(18)
        self.X = rng.normal(size=(240, 3))
        self.t = np.tile([0, 1], 120)
        probability = 1.0 / (1.0 + np.exp(-(-0.8 + self.X[:, 0] + 0.7 * self.t)))
        self.y = rng.binomial(1, probability)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_logistic_contracts_pickle_and_clone —— 核对 Logistic 学习器的接口、序列化和复制。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对 Logistic 学习器的接口、序列化和复制。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_logistic_contracts_pickle_and_clone(self):
        for estimator in (make_uplift_model("s", "logistic"), make_uplift_model("t", "logistic")):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_s_learner_has_only_additive_treatment_feature —— 确认 S 学习器仅添加处理主效应。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 S 学习器仅添加处理主效应。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_s_learner_has_only_additive_treatment_feature(self):
        model = make_uplift_model("s", "logistic").fit(self.X, self.t, self.y)
        self.assertEqual(
            model.model_["classifier"].coef_.shape, (1, self.X.shape[1] + 1)
        )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_preprocessing_is_fit_per_training_arm_only —— 确认预处理仅使用对应组的训练数据。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认预处理仅使用对应组的训练数据。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_preprocessing_is_fit_per_training_arm_only(self):
        X = np.array([[0.0], [np.nan], [2.0], [100.0], [np.nan], [104.0]])
        t = np.array([0, 0, 0, 1, 1, 1])
        y = np.array([0, 1, 0, 1, 0, 1])
        model = make_uplift_model("t", "logistic").fit(X, t, y)
        np.testing.assert_allclose(model.mu0_model_["imputer"].statistics_, [1.0])
        np.testing.assert_allclose(model.mu1_model_["imputer"].statistics_, [102.0])
        control_mean = model.mu0_model_["scaler"].mean_.copy()
        model.predict(np.array([[1e6], [np.nan]]))
        np.testing.assert_array_equal(model.mu0_model_["scaler"].mean_, control_mean)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_constant_outcome_arms_have_defined_positive_probabilities —— 确认单类别训练组仍有明确的正例概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认单类别训练组仍有明确的正例概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_constant_outcome_arms_have_defined_positive_probabilities(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        t = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        for estimator in (make_uplift_model("t", "logistic"), make_uplift_model("t", "xgboost")):
            estimator.fit(X, t, t)
            mu0, mu1 = estimator.predict_outcomes(X[:2])
            np.testing.assert_array_equal(mu0, [0, 0])
            np.testing.assert_array_equal(mu1, [1, 1])
            np.testing.assert_array_equal(estimator.predict(X[:2]), [1, 1])
        for outcome in (np.zeros(8), np.ones(8)):
            model = make_uplift_model("s", "logistic").fit(X, t, outcome)
            np.testing.assert_array_equal(model.predict(X[:2]), [0, 0])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_probability_selection_uses_class_label —— 确认正例概率按类别标签而非固定列位置提取。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认正例概率按类别标签而非固定列位置提取。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_probability_selection_uses_class_label(self):
        #-----------------------------------------------------------------------------------------------------------------------
        # 类：ReversedClasses —— 用反向类别顺序验证正例概率提取的测试模型。
        # 输入：
        #     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
        # 输出：
        #     实例：保存测试环境和断言的对象。
        # 说明：用反向类别顺序验证正例概率提取的测试模型。
        #-----------------------------------------------------------------------------------------------------------------------
        class ReversedClasses:
            classes_ = np.array([1, 0])

            #-----------------------------------------------------------------------------------------------------------------------
            # 函数：predict_proba —— 生成两类标签的预测概率。
            # 输入：
            #     self：当前对象，保存模型状态或测试上下文。
            #     X：处理前特征矩阵，行对应样本，列对应特征。
            # 输出：
            #     概率矩阵：每行依次包含类别 1 和类别 0 的概率。
            # 说明：生成两类标签的预测概率。
            #-----------------------------------------------------------------------------------------------------------------------
            def predict_proba(self, X):
                return np.tile([0.8, 0.2], (len(X), 1))

        np.testing.assert_allclose(
            _positive_probability(ReversedClasses(), self.X[:2]), 0.8
        )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_x_learner_cross_arm_targets_and_weight_direction —— 核对 X 学习器的跨组伪结果和加权方向。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对 X 学习器的跨组伪结果和加权方向。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_x_learner_cross_arm_targets_and_weight_direction(self):
        X = np.arange(16, dtype=float).reshape(8, 2)
        t = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        mu0, mu1 = _FixedProbabilityModel(0.2), _FixedProbabilityModel(0.8)
        tau0, tau1 = _RecordingRegressor(0.3), _RecordingRegressor(0.7)
        model = make_uplift_model("x", "xgboost", "xgboost")
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_validation_and_unfitted_behavior —— 核对输入检查及未拟合模型的错误行为。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对输入检查及未拟合模型的错误行为。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_validation_and_unfitted_behavior(self):
        with self.assertRaises(NotFittedError):
            make_uplift_model("t", "logistic").predict(self.X)
        invalid_cases = [
            (self.X, np.zeros(len(self.t)), self.y),
            (self.X, self.t, np.full(len(self.y), 2)),
            (self.X, self.t[:-1], self.y),
            (self.X, self.t, self.y.reshape(-1, 1)),
            (np.full(self.X.shape, np.inf), self.t, self.y),
        ]
        for X, t, y in invalid_cases:
            with self.assertRaises(ValueError):
                make_uplift_model("t", "logistic").fit(X, t, y)
        model = make_uplift_model("t", "logistic").fit(self.X, self.t, self.y)
        with self.assertRaises(ValueError):
            model.predict(self.X[:, :2])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_factory_and_tree_configuration_need_no_xgboost_import —— 确认模型创建与配置检查无需提前导入 XGBoost。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认模型创建与配置检查无需提前导入 XGBoost。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_factory_and_extra_combinations(self):
        for learner, outcome, effect, cls in [
            ("s", "logistic", None, SLearner), ("t", "logistic", None, TLearner),
            ("t", "xgboost", None, TLearner), ("x", "xgboost", "xgboost", XLearner),
            ("s", "xgboost", None, SLearner), ("x", "logistic", "linear", XLearner),
        ]:
            params = {"n_estimators": 3} if outcome == "xgboost" else {}
            model = make_uplift_model(learner, outcome, effect, random_state=31, **params)
            self.assertIsInstance(model, cls)
            self.assertEqual(clone(model).get_params(), model.get_params())
            model.fit(self.X, self.t, self.y)
            self.assertTrue(np.isfinite(model.predict(self.X)).all())
        for args in [("unknown", "logistic"), ("x", "logistic"), ("s", "logistic", "linear")]:
            with self.assertRaises(ValueError):
                make_uplift_model(*args)


if __name__ == "__main__":
    unittest.main()
