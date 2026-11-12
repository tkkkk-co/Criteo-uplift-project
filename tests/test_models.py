# 项目测试：验证模型接口及拟合逻辑。

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import NotFittedError

from criteo_uplift.models import (
    fit_s_learner, fit_t_learner, fit_x_learner, fit_uplift_model,
    predict_uplift, predict_outcomes, predict_positive_probability,
    uplift_model_parameters,
)


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

    def test_logistic_contracts_pickle_and_independent_fits(self):
        #-------------
        # 函数名：ModelTests.test_logistic_contracts_pickle_and_independent_fits
        # 功能：验证函数式训练结果支持序列化往返，重复训练不会共享已拟合模型。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for fit in (fit_s_learner, fit_t_learner):
            with self.subTest(function=fit.__name__):
                estimator = fit(self.X, self.t, self.y)
                mu0, mu1 = predict_outcomes(estimator, self.X[:11])
                self.assertEqual(mu0.shape, (11,))
                self.assertTrue(np.all((mu0 >= 0) & (mu0 <= 1)))
                self.assertTrue(np.all((mu1 >= 0) & (mu1 <= 1)))
                np.testing.assert_allclose(predict_uplift(estimator, self.X[:11]), mu1 - mu0)
                restored = pickle.loads(pickle.dumps(estimator))
                np.testing.assert_allclose(predict_uplift(restored, self.X[:11]), mu1 - mu0)
                other = fit(self.X, self.t, self.y)
                key = 'model' if estimator['learner'] == 's' else 'mu0_model'
                self.assertIsNot(estimator[key], other[key])
                self.assertEqual(estimator['parameters'], other['parameters'])
                self.assertEqual(estimator['overlap']['status'], 'not_checked')

    def test_s_learner_has_only_additive_treatment_feature(self):
        #-------------
        # 函数名：ModelTests.test_s_learner_has_only_additive_treatment_feature
        # 功能：验证 S-Learner 仅将 T 作为附加输入特征，并分别预测 T=0 和 T=1 的潜在结果。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        model = fit_s_learner(self.X, self.t, self.y)
        self.assertEqual(
            model['model']["classifier"].coef_.shape, (1, self.X.shape[1] + 1)
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
        model = fit_t_learner(X, t, y)
        np.testing.assert_allclose(model['mu0_model']["imputer"].statistics_, [1.0])
        np.testing.assert_allclose(model['mu1_model']["imputer"].statistics_, [102.0])
        control_mean = model['mu0_model']["scaler"].mean_.copy()
        predict_uplift(model, np.array([[1e6], [np.nan]]))
        np.testing.assert_array_equal(model['mu0_model']["scaler"].mean_, control_mean)

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
        for name in ('logistic', 'xgboost'):
            estimator = fit_t_learner(X, t, t, outcome_model=name)
            mu0, mu1 = predict_outcomes(estimator, X[:2])
            np.testing.assert_array_equal(mu0, [0, 0])
            np.testing.assert_array_equal(mu1, [1, 1])
            np.testing.assert_array_equal(predict_uplift(estimator, X[:2]), [1, 1])
        for outcome in (np.zeros(8), np.ones(8)):
            model = fit_s_learner(X, t, outcome)
            np.testing.assert_array_equal(predict_uplift(model, X[:2]), [0, 0])

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
            predict_positive_probability(ReversedClasses(), self.X[:2]), 0.8
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
        with patch('criteo_uplift.models.make_classifier_model', side_effect=[mu0, mu1]), patch(
            'criteo_uplift.models.make_regression_model', side_effect=[tau0, tau1]
        ):
            model = fit_x_learner(X, t, y)
        np.testing.assert_array_equal(mu0.fit_X, X[t == 0])
        np.testing.assert_array_equal(mu1.fit_X, X[t == 1])
        np.testing.assert_allclose(tau0.fit_y, 0.8 - y[t == 0])
        np.testing.assert_allclose(tau1.fit_y, y[t == 1] - 0.2)
        self.assertEqual(model['treatment_rate'], 0.25)
        np.testing.assert_allclose(predict_uplift(model, X[:2]), 0.25 * 0.3 + 0.75 * 0.7)
        np.testing.assert_allclose(
            predict_outcomes(model, X[:2]), [[0.2, 0.2], [0.8, 0.8]]
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
            predict_uplift({}, self.X)
        invalid_cases = [
            (self.X, np.zeros(len(self.t)), self.y),
            (self.X, self.t, np.full(len(self.y), 2)),
            (self.X, self.t[:-1], self.y),
            (self.X, self.t, self.y.reshape(-1, 1)),
            (np.full(self.X.shape, np.inf), self.t, self.y),
        ]
        for X, t, y in invalid_cases:
            with self.assertRaises(ValueError):
                fit_t_learner(X, t, y)
        model = fit_t_learner(self.X, self.t, self.y)
        with self.assertRaises(ValueError):
            predict_uplift(model, self.X[:, :2])

    def test_factory_and_extra_combinations(self):
        #-------------
        # 函数名：ModelTests.test_factory_and_extra_combinations
        # 功能：验证模型工厂可以组合 S/T/X Learner 与支持的 Outcome/Effect Base Model，而不局限于正式候选组合。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for learner, outcome, effect in [
            ("s", "logistic", None), ("t", "logistic", None),
            ("t", "xgboost", None), ("x", "xgboost", "xgboost"),
            ("s", "xgboost", None), ("x", "logistic", "linear"),
        ]:
            params = {"n_estimators": 3} if outcome == "xgboost" else {}
            model = fit_uplift_model(self.X, self.t, self.y, learner, outcome, effect, random_state=31, **params)
            self.assertEqual(model['learner'], learner)
            self.assertEqual(model['parameters']['outcome_params']['random_state'], 31)
            self.assertTrue(np.isfinite(predict_uplift(model, self.X)).all())
        for args in [("unknown", "logistic"), ("x", "logistic"), ("s", "logistic", "linear")]:
            with self.assertRaises(ValueError):
                uplift_model_parameters(*args)


class FunctionalModelTests(unittest.TestCase):
    #-------------
    # 验证函数式接口新增的共享诊断、配置透传和模型保存契约。
    #-------------
    def test_shared_overlap_and_feature_order_validation(self):
        #-------------
        # 功能：共享诊断必须来自同一训练数据，预测阶段不重复诊断。
        # 输入参数：
        #     - self：测试实例。
        # 输出参数：
        #     - 无；检查共享对象、来源匹配及列顺序。
        #-------------
        import pandas as pd
        from criteo_uplift.diagnostics import diagnose_overlap
        x = pd.DataFrame({'a': np.arange(24.), 'b': np.tile([0., 1., 2.], 8)})
        t, y = np.tile([0, 1], 12), np.tile([0, 0, 1, 1], 6)
        summary = diagnose_overlap(x, t, bins=2, min_arm_count=1)['summary']
        models = [fit_s_learner(x, t, y, overlap=summary), fit_t_learner(x, t, y, overlap=summary),
                  fit_x_learner(x, t, y, 'logistic', 'linear', overlap=summary)]
        with patch('criteo_uplift.diagnostics.diagnose_overlap', side_effect=AssertionError('预测时不可诊断')):
            for model in models:
                self.assertIs(model['overlap'], summary)
                self.assertEqual(predict_uplift(model, x).shape, (24,))
                with self.assertRaisesRegex(ValueError, '顺序'):
                    predict_uplift(model, x[['b', 'a']])
        with self.assertRaisesRegex(ValueError, '不匹配'):
            fit_t_learner(x + 1, t, y, overlap=summary)
        with self.assertRaisesRegex(ValueError, '不匹配'):
            fit_t_learner(x, 1-t, y, overlap=summary)

    def test_fit_counts_and_parameter_overrides(self):
        #-------------
        # 功能：S 只训练一个分类器，T 两个分类器独立；阶段参数覆盖公共参数。
        # 输入参数：
        #     - self：测试实例。
        # 输出参数：
        #     - 无；检查拟合输入及模型配置。
        #-------------
        x, t, y = np.arange(24.).reshape(8, 3), np.tile([0, 1], 4), np.tile([0, 0, 1, 1], 2)
        with patch('criteo_uplift.models.make_classifier_model', return_value=_FixedProbabilityModel(.4)) as factory:
            s = fit_s_learner(x, t, y, C=.5, outcome_params={'C': 2.})
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.kwargs['C'], 2.)
            np.testing.assert_array_equal(s['model'].fit_X, np.column_stack((x, t)))
        with patch('criteo_uplift.models.make_classifier_model', side_effect=[_FixedProbabilityModel(.2), _FixedProbabilityModel(.4)]) as factory:
            fitted = fit_t_learner(x, t, y)
            self.assertEqual(factory.call_count, 2)
            self.assertIsNot(fitted['mu0_model'], fitted['mu1_model'])
            np.testing.assert_array_equal(fitted['mu0_model'].fit_X, x[t == 0])
            np.testing.assert_array_equal(fitted['mu1_model'].fit_X, x[t == 1])
        fitted = fit_x_learner(x, t, y, n_estimators=3, outcome_params={'n_estimators': 4},
                               effect_params={'n_estimators': 2})
        self.assertEqual(fitted['mu0_model'].get_params()['n_estimators'], 4)
        self.assertEqual(fitted['tau0_model'].get_params()['n_estimators'], 2)
        self.assertEqual(fitted['tau0_model'].get_params()['objective'], 'reg:squarederror')
        self.assertEqual(fitted['mu0_model'].get_params()['objective'], 'binary:logistic')

    def test_joblib_roundtrip_all_candidates_and_nan(self):
        #-------------
        # 功能：四个候选模型都能保存重载，并安全预测缺失特征。
        # 输入参数：
        #     - self：测试实例。
        # 输出参数：
        #     - 无；检查序列化前后逐行相等。
        #-------------
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import joblib
        from criteo_uplift.training_validation import MODEL_CANDIDATES
        rng = np.random.default_rng(16)
        x = rng.normal(size=(80, 3))
        x[::5, 0] = np.nan
        x[:, 2] = np.nan
        t, y = np.tile([0, 1], 40), np.tile([0, 0, 1, 1], 20)
        with TemporaryDirectory() as folder:
            for candidate in MODEL_CANDIDATES:
                config = {k:v for k,v in candidate.items() if k != 'name'}
                params = {'n_estimators': 3} if candidate['outcome_model'] == 'xgboost' else {}
                model = fit_uplift_model(x, t, y, **config, **params)
                before = predict_uplift(model, x)
                path = Path(folder) / 'model.joblib'
                joblib.dump(model, path)
                np.testing.assert_array_equal(before, predict_uplift(joblib.load(path), x))
                self.assertTrue(np.isfinite(before).all())


if __name__ == "__main__":
    unittest.main()
