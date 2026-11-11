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

    def __init__(self, probability):
        #-------------
        # 函数名：_FixedProbabilityModel.__init__
        # 功能：保存测试分类器的固定正类概率。
        # 输入参数：
        #     - probability：模型对每条样本输出的正类概率。
        # 输出参数：
        #     - 无；保存 probability 属性。
        #-------------
        self.probability = probability

    def fit(self, X, y):
        #-------------
        # 函数名：_FixedProbabilityModel.fit
        # 功能：记录传入的特征和标签，模拟一个已经拟合的分类器。
        # 输入参数：
        #     - X：用于拟合的测试特征矩阵。
        #     - y：用于拟合的二元标签。
        # 输出参数：
        #     - 当前测试分类器实例。
        #-------------
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    def predict_proba(self, X):
        #-------------
        # 函数名：_FixedProbabilityModel.predict_proba
        # 功能：为每条输入样本返回固定的负类和正类概率。
        # 输入参数：
        #     - X：需要输出概率的测试特征矩阵。
        # 输出参数：
        #     - n_samples × 2 的类别概率矩阵。
        #-------------
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
    def __init__(self, prediction):
        #-------------
        # 函数名：_RecordingRegressor.__init__
        # 功能：保存测试回归器的固定预测值。
        # 输入参数：
        #     - prediction：每条样本使用的固定回归预测值。
        # 输出参数：
        #     - 无；保存 prediction 属性。
        #-------------
        self.prediction = prediction

    def fit(self, X, y):
        #-------------
        # 函数名：_RecordingRegressor.fit
        # 功能：记录效应回归器接收的训练特征和伪效应目标。
        # 输入参数：
        #     - X：效应模型训练特征矩阵。
        #     - y：效应模型训练目标，即伪处理效应。
        # 输出参数：
        #     - 当前测试回归器实例。
        #-------------
        self.fit_X = np.array(X)
        self.fit_y = np.array(y)
        return self

    def predict(self, X):
        #-------------
        # 函数名：_RecordingRegressor.predict
        # 功能：对每条输入样本返回预设的固定效应预测。
        # 输入参数：
        #     - X：需要预测效应的特征矩阵。
        # 输出参数：
        #     - 长度与输入样本数相同的回归预测数组。
        #-------------
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
    def setUp(self):
        #-------------
        # 函数名：ModelTests.setUp
        # 功能：生成模型单元测试共用的合成 X、T、Y 数据及随机状态。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化或清理测试实例状态。
        #-------------
        rng = np.random.default_rng(18)
        self.X = rng.normal(size=(240, 3))
        self.t = np.tile([0, 1], 120)
        probability = 1.0 / (1.0 + np.exp(-(-0.8 + self.X[:, 0] + 0.7 * self.t)))
        self.y = rng.binomial(1, probability)

    def test_logistic_contracts_pickle_and_clone(self):
        #-------------
        # 函数名：ModelTests.test_logistic_contracts_pickle_and_clone
        # 功能：验证 Logistic Uplift Learner 符合 sklearn estimator 接口，并支持 clone 和 joblib 序列化往返。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_s_learner_has_only_additive_treatment_feature(self):
        #-------------
        # 函数名：ModelTests.test_s_learner_has_only_additive_treatment_feature
        # 功能：验证 S-Learner 仅将 T 作为附加输入特征，并分别预测 T=0 和 T=1 的潜在结果。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        model = make_uplift_model("s", "logistic").fit(self.X, self.t, self.y)
        self.assertEqual(
            model.model_["classifier"].coef_.shape, (1, self.X.shape[1] + 1)
        )

    def test_preprocessing_is_fit_per_training_arm_only(self):
        #-------------
        # 函数名：ModelTests.test_preprocessing_is_fit_per_training_arm_only
        # 功能：验证缺失值插补和标准化只在各自训练数据或训练处理臂上拟合，不读取验证信息。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        X = np.array([[0.0], [np.nan], [2.0], [100.0], [np.nan], [104.0]])
        t = np.array([0, 0, 0, 1, 1, 1])
        y = np.array([0, 1, 0, 1, 0, 1])
        model = make_uplift_model("t", "logistic").fit(X, t, y)
        np.testing.assert_allclose(model.mu0_model_["imputer"].statistics_, [1.0])
        np.testing.assert_allclose(model.mu1_model_["imputer"].statistics_, [102.0])
        control_mean = model.mu0_model_["scaler"].mean_.copy()
        model.predict(np.array([[1e6], [np.nan]]))
        np.testing.assert_array_equal(model.mu0_model_["scaler"].mean_, control_mean)

    def test_constant_outcome_arms_have_defined_positive_probabilities(self):
        #-------------
        # 函数名：ModelTests.test_constant_outcome_arms_have_defined_positive_probabilities
        # 功能：验证某个处理臂的结果只有单一类别时，常数概率分类器仍能返回定义良好的正类概率。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_probability_selection_uses_class_label(self):
        #-------------
        # 函数名：ModelTests.test_probability_selection_uses_class_label
        # 功能：验证正类概率按 classes_ 中的类别标签 1 定位，而不是假定概率矩阵固定列顺序。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

            def predict_proba(self, X):
                #-------------
                # 函数名：ModelTests.test_probability_selection_uses_class_label.ReversedClasses.predict_proba
                # 功能：构造类别顺序为 [1,0] 的概率矩阵，用于验证正类概率列按标签定位。
                # 输入参数：
                #     - X：需要返回类别概率的特征矩阵。
                # 输出参数：
                #     - 列顺序对应 classes_=[1,0] 的概率矩阵。
                #-------------
                return np.tile([0.8, 0.2], (len(X), 1))

        np.testing.assert_allclose(
            _positive_probability(ReversedClasses(), self.X[:2]), 0.8
        )

    def test_x_learner_cross_arm_targets_and_weight_direction(self):
        #-------------
        # 函数名：ModelTests.test_x_learner_cross_arm_targets_and_weight_direction
        # 功能：验证 X-Learner 两组跨臂伪效应目标、Effect Model 训练输入和按训练处理比例加权的方向。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_validation_and_unfitted_behavior(self):
        #-------------
        # 函数名：ModelTests.test_validation_and_unfitted_behavior
        # 功能：验证模型拒绝未拟合预测、错误特征数量、错误列顺序和不合法训练输入。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_factory_and_extra_combinations(self):
        #-------------
        # 函数名：ModelTests.test_factory_and_extra_combinations
        # 功能：验证模型工厂可以组合 S/T/X Learner 与支持的 Outcome/Effect Base Model，而不局限于正式候选组合。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
