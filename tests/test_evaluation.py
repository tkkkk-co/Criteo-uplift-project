# 项目测试：验证效果估计和策略评价。

import math
from io import StringIO
import pandas as pd
import unittest

import numpy as np

from criteo_uplift.metrics import _prefix_values, ranking_metrics, ranking_confidence_intervals, uplift_curve
from criteo_uplift.evaluation import _normal_pvalue, ate_summary_from_audit, ate_summary, calibration_table, evaluate_policies


#-----------------------------------------------------------------------------------------------------------------------
# 类：EvaluationTests —— 验证效果估计和策略评价。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证效果估计和策略评价。
#-----------------------------------------------------------------------------------------------------------------------
class EvaluationTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
    """
    函数名：EvaluationTests.setUp
    功能：准备 ATE、HT 曲线和 Policy Evaluation 测试共用的二元样本及策略分数。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；初始化或清理测试实例状态。
    """
        self.y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        self.t = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        self.scores = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_difference_in_means_and_pooled_null —— 核对平均效果估计及零假设下的检验。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对平均效果估计及零假设下的检验。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_difference_in_means_and_pooled_null(self):
    """
    函数名：EvaluationTests.test_difference_in_means_and_pooled_null
    功能：验证 ATE 差均值估计、标准误和 pooled 零假设检验与手工公式一致。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        result = ate_summary(self.y, self.t)
        self.assertEqual(result["ate"], 0.5)
        self.assertEqual(result["relative_uplift"], 2.0)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(0.125))
        self.assertAlmostEqual(result["p_value"], math.erfc(1.0))
        self.assertLess(result["ci_low"], result["ate"])
        self.assertGreater(result["ci_high"], result["ate"])

    def test_ate_row_and_audit_paths_agree_for_every_statistic(self):
    """
    函数名：EvaluationTests.test_ate_row_and_audit_paths_agree_for_every_statistic
    功能：验证逐行 ATE 与 Audit 计数 ATE 的估计、标准误、置信区间和 p 值字段一致。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        for y, t in ((self.y, self.t), (np.array([1, 0, 1]), np.array([0, 1, 1])),
                     (np.zeros(8), self.t), (np.ones(8), self.t)):
            audit = {"groups": {str(arm): {"n": int((t == arm).sum()),
                     "conversion": {"positive": int(y[t == arm].sum())}} for arm in (0, 1)}}
            actual, expected = ate_summary_from_audit(audit, "conversion"), ate_summary(y, t)
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                if isinstance(expected[key], str) or expected[key] is None:
                    self.assertEqual(actual[key], expected[key])
                else:
                    np.testing.assert_allclose(actual[key], expected[key], equal_nan=True, rtol=0, atol=0)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_aggregate_ate_rejects_invalid_counts —— 确认汇总 ATE 拒绝不合法的人数和事件数。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认汇总 ATE 拒绝不合法的人数和事件数。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_normal_tail_retains_subnormal_probability_and_log_beyond_float_range(self):
    """
    函数名：EvaluationTests.test_normal_tail_retains_subnormal_probability_and_log_beyond_float_range
    功能：验证极端正态尾部概率在浮点下溢附近仍保留可用的 log p 值和文本表示。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        # z=38 的双侧尾概率仍可用 subnormal float 表达，不能提前截为零。
        p, log_p, text = _normal_pvalue(38.0)
        self.assertGreater(p, 0.0)
        self.assertLess(p, np.finfo(float).tiny)
        self.assertAlmostEqual(math.log(p), log_p, places=7)
        # z=50 超出 float 范围；用 Mills 上下界独立核对稳定对数概率。
        p, log_p, text = _normal_pvalue(50.0)
        upper = math.log(2) - 50**2 / 2 - .5 * math.log(2 * math.pi) - math.log(50)
        lower = upper + math.log(50**2 / (50**2 + 1))
        self.assertEqual(p, 0.0)
        self.assertTrue(math.isfinite(log_p))
        self.assertLess(log_p, upper)
        self.assertGreater(log_p, lower)
        self.assertRegex(text, r"^[1-9]\.\d{3}e-\d+$")
        self.assertEqual(_normal_pvalue(-50.0), (p, log_p, text))
        self.assertEqual(_normal_pvalue(0.0), (1.0, 0.0, '1.000e+00'))

    def test_ate_extreme_pvalues_preserve_test_and_estimates(self):
    """
    函数名：EvaluationTests.test_ate_extreme_pvalues_preserve_test_and_estimates
    功能：验证极端 p 值场景下检验结果不会改变 ATE 点估计及其区间。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        y = np.r_[np.ones(8000), np.zeros(2000), np.ones(1000), np.zeros(9000)]
        t = np.repeat([1, 0], 10000)
        row = ate_summary(y, t)
        audit = {'groups': {
            str(arm): {'n': 10000, 'conversion': {'positive': int(y[t == arm].sum())}}
            for arm in (0, 1)
        }}
        aggregate = ate_summary_from_audit(audit, 'conversion')
        self.assertAlmostEqual(row['ate'], .7)
        self.assertEqual(row['p_value'], 0.0)
        self.assertEqual(aggregate['p_value'], 0.0)
        self.assertEqual(row['log_p_value'], aggregate['log_p_value'])
        self.assertEqual(row['p_value_text'], aggregate['p_value_text'])
        saved = StringIO(pd.DataFrame([row]).to_csv(index=False))
        restored = pd.read_csv(saved, dtype={'p_value_text': 'string'})
        self.assertEqual(restored.loc[0, 'p_value_text'], row['p_value_text'])
        self.assertTrue(math.isfinite(row['log_p_value']))
        self.assertAlmostEqual(row['standard_error'], math.sqrt(y[t == 1].var(ddof=1)/10000 + y[t == 0].var(ddof=1)/10000))
        self.assertAlmostEqual(aggregate['standard_error'], math.sqrt(.8*.2/9999 + .1*.9/9999))
        constant = ate_summary(np.zeros(8), self.t)
        self.assertEqual(constant['log_p_value'], 0.0)
        self.assertTrue(np.isnan(ate_summary(np.r_[1., 0., 1.], np.r_[0, 1, 1])['standard_error']))

    def test_aggregate_ate_rejects_invalid_counts(self):
    """
    函数名：EvaluationTests.test_aggregate_ate_rejects_invalid_counts
    功能：验证 ATE 计数聚合函数拒绝非法人数、正例数和非整数计数。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        for n, positive in ((0, 0), (2, 3), (2, -1), (2.5, 1), (True, 1)):
            audit = {'groups': {
                '0': {'n': n, 'conversion': {'positive': positive}},
                '1': {'n': 3, 'conversion': {'positive': 1}},
            }}
            with self.subTest(n=n, positive=positive), self.assertRaises(ValueError):
                ate_summary_from_audit(audit, 'conversion')

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_constant_outcomes_and_missing_control_response —— 核对结果恒定及对照组无正例的边界行为。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对结果恒定及对照组无正例的边界行为。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_constant_outcomes_and_missing_control_response(self):
    """
    函数名：EvaluationTests.test_constant_outcomes_and_missing_control_response
    功能：验证常量结果和某一处理臂缺失正例时 ATE 的边界计算行为。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        result = ate_summary(np.zeros(8), self.t)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["standard_error"], 0.0)
        self.assertTrue(np.isnan(result["relative_uplift"]))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_ht_curve_preserves_fixed_assignment_probability —— 确认 HT 曲线使用固定的处理分配概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 HT 曲线使用固定的处理分配概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_ht_curve_preserves_fixed_assignment_probability(self):
    """
    函数名：EvaluationTests.test_ht_curve_preserves_fixed_assignment_probability
    功能：验证 HT 曲线始终使用调用方传入的固定处理分配概率。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        curve = uplift_curve(self.y, self.t, self.scores, e=0.8)
        expected_z = self.t * self.y / 0.8 - (1 - self.t) * self.y / 0.2
        np.testing.assert_allclose(curve.gain_ht, np.r_[0.0, expected_z.cumsum()] / 8)
        self.assertNotAlmostEqual(
            curve.gain_ht.iloc[-1], ate_summary(self.y, self.t)["ate"]
        )
        self.assertTrue(np.isnan(curve.qini_traditional.iloc[2]))
        self.assertAlmostEqual(curve.qini_traditional.iloc[3], -1 / 8)
        self.assertAlmostEqual(curve.uplift.iloc[3], -0.5)
        self.assertAlmostEqual(curve.lift.iloc[-1], 1)
        self.assertTrue(np.isnan(curve.uplift.iloc[2]))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_area_uses_full_curve_and_centering —— 核对完整曲线面积及随机基线扣除。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对完整曲线面积及随机基线扣除。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_area_uses_full_curve_and_centering(self):
    """
    函数名：EvaluationTests.test_area_uses_full_curve_and_centering
    功能：验证曲线积分使用完整前缀分辨率，并按终点随机基线进行中心化。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        curve = uplift_curve(self.y, self.t, self.scores, e=0.5)
        metrics = ranking_metrics(self.y, self.t, self.scores, e=0.5)
        expected = sum(
            (curve.gain_ht.iloc[i] + curve.gain_ht.iloc[i + 1]) / 16 for i in range(8)
        )
        self.assertAlmostEqual(metrics["auuc_ht"], expected)
        self.assertAlmostEqual(metrics["qini_ht"], expected - metrics["ate_ht"] / 2)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_traditional_qini_bridges_initial_undefined_prefix —— 核对传统 Qini 对开头未定义区间的处理。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对传统 Qini 对开头未定义区间的处理。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_traditional_qini_bridges_initial_undefined_prefix(self):
    """
    函数名：EvaluationTests.test_traditional_qini_bridges_initial_undefined_prefix
    功能：验证传统 Qini 在初始前缀未同时出现两组时采用规定的积分连接方式。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        y, t = [1, 0, 0, 1], [1, 1, 0, 0]
        scores = [4, 3, 2, 1]
        curve = uplift_curve(y, t, scores, e=0.5)
        metrics = ranking_metrics(y, t, scores, e=0.5)
        # 传统 Qini 的有效点为 (3/4, 1/4)、(1, 0)，并加入原点。
        # 两段梯形面积分别为 3/32 和 1/32，随机基线面积为零。
        self.assertTrue(curve.qini_traditional.iloc[:3].isna().all())
        self.assertEqual(curve.qini_traditional.iloc[-1], 0)
        self.assertEqual(metrics["qini_score_traditional"], 1 / 8)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_traditional_qini_centers_its_own_full_coverage_endpoint —— 确认传统 Qini 使用自身终点构造随机基线。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认传统 Qini 使用自身终点构造随机基线。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_traditional_qini_centers_its_own_full_coverage_endpoint(self):
    """
    函数名：EvaluationTests.test_traditional_qini_centers_its_own_full_coverage_endpoint
    功能：验证传统 Qini 使用自身全覆盖终点计算随机基线面积。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        y, t = [1, 0, 0, 0], [1, 1, 0, 0]
        scores = [4, 3, 2, 1]
        curve = uplift_curve(y, t, scores, e=0.6)
        metrics = ranking_metrics(y, t, scores, e=0.6)
        # 传统 Qini 终点为 1/4，原始面积为 3/32 + 1/16 = 5/32。
        # 扣除随机基线的三角形面积 1/8，得到 1/32。
        self.assertEqual(curve.qini_traditional.iloc[-1], 1 / 4)
        self.assertEqual(metrics["qini_score_traditional"], 1 / 32)
        self.assertNotEqual(metrics["qini_score_traditional"], metrics["qini_ht"])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_traditional_score_is_not_changed_by_ht_assignment_probability —— 确认传统 Qini 不受 HT 分配概率影响。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认传统 Qini 不受 HT 分配概率影响。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_traditional_score_is_not_changed_by_ht_assignment_probability(self):
    """
    函数名：EvaluationTests.test_traditional_score_is_not_changed_by_ht_assignment_probability
    功能：验证传统 Qini 排序分数不受 HT 处理分配概率参数影响。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        a = ranking_metrics(self.y, self.t, self.scores, e=0.5)
        b = ranking_metrics(self.y, self.t, self.scores, e=0.8)
        self.assertEqual(a["qini_score_traditional"], b["qini_score_traditional"])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_tie_break_is_seeded_and_does_not_use_row_order —— 确认同分排序可复现且不直接使用原始行序。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认同分排序可复现且不直接使用原始行序。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_tie_break_is_seeded_and_does_not_use_row_order(self):
    """
    函数名：EvaluationTests.test_tie_break_is_seeded_and_does_not_use_row_order
    功能：验证相同分数的排序由独立随机种子决定，而不依赖输入行顺序。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        n = 100
        t = np.r_[np.ones(n // 2), np.zeros(n // 2)]
        y = np.ones(n)
        a = uplift_curve(y, t, np.ones(n), e=0.5, seed=42)
        b = uplift_curve(y, t, np.ones(n), e=0.5, seed=42)
        c = uplift_curve(y, t, np.ones(n), e=0.5, seed=43)
        np.testing.assert_array_equal(a.gain_ht, b.gain_ht)
        self.assertFalse(np.array_equal(a.gain_ht, c.gain_ht))
        self.assertGreater(a.n_control.iloc[10], 0)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_weighted_prefix_reselects_boundary —— 核对重抽样后预算边界的重新选择。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对重抽样后预算边界的重新选择。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_weighted_prefix_reselects_boundary(self):
    """
    函数名：EvaluationTests.test_weighted_prefix_reselects_boundary
    功能：验证 Bootstrap 权重压缩表示在 top-k 边界处正确重新选择部分样本。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        # 展开重复次数后的排序样本为 [3, 3, -2, -2, -2, 7]。
        z = np.array([10.0, 3.0, -2.0, 7.0])
        counts = np.array([0, 2, 3, 1])
        k = np.array([0, 1, 2, 3, 5, 6])
        expected = np.r_[0, np.cumsum(np.repeat(z, counts))][k]
        np.testing.assert_array_equal(_prefix_values(z, counts, k), expected)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_same_score_paired_difference_is_exactly_zero —— 确认相同分数的配对差异及区间为零。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认相同分数的配对差异及区间为零。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_same_score_paired_difference_is_exactly_zero(self):
    """
    函数名：EvaluationTests.test_same_score_paired_difference_is_exactly_zero
    功能：验证两个完全相同的策略分数在配对比较中的增量差值严格为零。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        result = evaluate_policies(
            self.y,
            self.t,
            {"propensity": self.scores, "T": self.scores},
            e=0.5,
            budgets=[0.25, 0.5, 1.0],
            bootstrap=40,
            seed=19,
        )
        same = result[result.policy == "T"]
        np.testing.assert_array_equal(same.vs_propensity_per_100k, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_low, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_high, 0)
        full = result[result.budget == 1]
        np.testing.assert_allclose(full.incremental_per_100k, 50_000)
        np.testing.assert_allclose(full.vs_random_per_100k, 0, atol=1e-10)
        np.testing.assert_allclose(full.vs_random_ci_low, 0, atol=1e-10)
        np.testing.assert_allclose(full.vs_random_ci_high, 0, atol=1e-10)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_bootstrap_is_reproducible_and_point_estimates_unchanged —— 确认重抽样可复现且不改变点估计。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认重抽样可复现且不改变点估计。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_bootstrap_is_reproducible_and_point_estimates_unchanged(self):
    """
    函数名：EvaluationTests.test_bootstrap_is_reproducible_and_point_estimates_unchanged
    功能：验证 Policy Bootstrap 在相同随机种子下可复现，且不改变原始点估计。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        kwargs = dict(
            y=self.y,
            t=self.t,
            score_dict={"S": self.scores},
            e=0.5,
            budgets=[0.5],
            seed=77,
        )
        a = evaluate_policies(**kwargs, bootstrap=30)
        b = evaluate_policies(**kwargs, bootstrap=30)
        point = evaluate_policies(**kwargs, bootstrap=0)
        np.testing.assert_array_equal(a.ci_low, b.ci_low)
        np.testing.assert_array_equal(a.ci_high, b.ci_high)
        np.testing.assert_array_equal(
            a.incremental_per_100k, point.incremental_per_100k
        )
        self.assertTrue(point.ci_low.isna().all())
        self.assertTrue(point.vs_propensity_per_100k.isna().all())

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_budget_rounding_and_analytic_random —— 核对预算取整及解析随机策略收益。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对预算取整及解析随机策略收益。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_budget_rounding_and_analytic_random(self):
    """
    函数名：EvaluationTests.test_budget_rounding_and_analytic_random
    功能：验证预算比例向下取整得到 top-k 人数，并正确计算随机策略解析期望。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        result = evaluate_policies(
            self.y, self.t, {"S": self.scores}, e=0.5, budgets=[0.01, 0.3], bootstrap=0
        )
        random = result[result.policy == "random"]
        np.testing.assert_array_equal(random.n_selected, [0, 2])
        np.testing.assert_array_equal(random.selected_fraction, [0, 0.25])
        np.testing.assert_array_equal(random.incremental_per_100k, [0, 12_500])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_paired_difference_matches_direct_resampling —— 用显式重抽样核对配对差异区间。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：用显式重抽样核对配对差异区间。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_paired_difference_matches_direct_resampling(self):
    """
    函数名：EvaluationTests.test_paired_difference_matches_direct_resampling
    功能：验证优化后的配对 Bootstrap 差值区间与显式直接重抽样结果一致。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        # 通过显式展开抽样行，独立核对优化后的重抽样计算。
        budgets = np.array([0.25, 0.5])
        seed, replicates = 15, 30
        result = evaluate_policies(
            self.y,
            self.t,
            {"S": self.scores},
            e=0.5,
            budgets=budgets,
            bootstrap=replicates,
            seed=seed,
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
        z = 2 * self.t * self.y - 2 * (1 - self.t) * self.y
        arm1, arm0 = np.flatnonzero(self.t == 1), np.flatnonzero(self.t == 0)
        delta = []
        for _ in range(replicates):
            sample = np.r_[
                arm1[rng.integers(len(arm1), size=len(arm1))],
                arm0[rng.integers(len(arm0), size=len(arm0))],
            ]
            sample = sample[np.argsort(-self.scores[sample], kind="stable")]
            gain = np.r_[0, np.cumsum(z[sample])][(budgets * 8).astype(int)] / 8
            delta.append((gain - budgets * z[sample].mean()) * 100_000)
        expected = np.quantile(delta, [0.025, 0.975], axis=0)
        selected = result[result.policy == "S"]
        np.testing.assert_allclose(selected.vs_random_ci_low, expected[0])
        np.testing.assert_allclose(selected.vs_random_ci_high, expected[1])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_calibration_bins_cover_data_and_missing_arms_are_nan —— 核对校准分箱覆盖及缺少组别时的缺失结果。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对校准分箱覆盖及缺少组别时的缺失结果。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_calibration_bins_cover_data_and_missing_arms_are_nan(self):
    """
    函数名：EvaluationTests.test_calibration_bins_cover_data_and_missing_arms_are_nan
    功能：验证校准分箱完整覆盖样本，缺少某个处理臂的分箱保留 NaN。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        bins = calibration_table(self.y, self.t, self.scores, n_bins=4)
        self.assertEqual(bins.n.sum(), len(self.y))
        self.assertTrue(bins.predicted_cate.is_monotonic_increasing)
        self.assertTrue(bins.observed_ate.isna().all())
        single = calibration_table(self.y, self.t, self.scores, n_bins=1)
        self.assertEqual(single.observed_ate.iloc[0], 0.5)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_inputs_fail_explicitly —— 确认评价函数明确拒绝非法输入。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认评价函数明确拒绝非法输入。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_inputs_fail_explicitly(self):
    """
    函数名：EvaluationTests.test_invalid_inputs_fail_explicitly
    功能：验证 ATE、Policy Evaluation 和校准函数对非法输入明确报错。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        with self.assertRaises(ValueError):
            ate_summary([1, 0], [1, 1])
        with self.assertRaises(ValueError):
            ate_summary([1, np.nan], [1, 0])
        with self.assertRaises(ValueError):
            uplift_curve(self.y, self.t, self.scores, e=1)
        with self.assertRaises(ValueError):
            ranking_metrics(self.y, self.t, [1, 2], e=0.5)
        with self.assertRaises(ValueError):
            evaluate_policies(
                self.y, self.t, {"random": self.scores}, e=0.5, bootstrap=0
            )
        with self.assertRaises(ValueError):
            evaluate_policies(self.y, self.t, {}, e=0.5, budgets=[1.2], bootstrap=0)


#-----------------------------------------------------------------------------------------------------------------------
# 类：ExtendedMetricTests —— 验证效果估计和策略评价。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证效果估计和策略评价。
#-----------------------------------------------------------------------------------------------------------------------
class ExtendedMetricTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
    """
    函数名：ExtendedMetricTests.setUp
    功能：准备扩展排序指标测试使用的八行手工数据和评分向量。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；初始化或清理测试实例状态。
    """
        self.t = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        self.s = np.arange(8, 0, -1)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_full_eight_row_hand_example —— 用八行手算样例核对各排序指标。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：用八行手算样例核对各排序指标。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_full_eight_row_hand_example(self):
    """
    函数名：ExtendedMetricTests.test_full_eight_row_hand_example
    功能：使用八行手工样本核对 uplift curve、HT gain 和 Qini 指标的完整计算结果。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        c = uplift_curve(self.y, self.t, self.s, 0.5)
        np.testing.assert_allclose(
            c.cumulative_lift.iloc[2:], [1, 1, 1, 2 / 3, 1 / 3, 5 / 12, 1 / 2]
        )
        np.testing.assert_allclose(
            c.gain_plugin.iloc[2:], [1 / 4, 3 / 8, 1 / 2, 5 / 12, 1 / 4, 35 / 96, 1 / 2]
        )
        np.testing.assert_allclose(
            c.qini_traditional.iloc[2:],
            [1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8, 5 / 24, 1 / 4],
        )
        m = ranking_metrics(self.y, self.t, self.s, 0.5)
        for name, expected in {
            'auuc_plugin': 81 / 256,
            'auuc_plugin_centered': 17 / 256,
            'qini_score_traditional': 19 / 384,
            'auuc_ht': 3 / 8,
            'qini_ht': 1 / 8,
        }.items():
            self.assertAlmostEqual(m[name], expected)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_qini_keeps_defined_control_only_zero_nodes —— 确认 Qini 保留仅有对照组时的零值前缀。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 Qini 保留仅有对照组时的零值前缀。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_qini_keeps_defined_control_only_zero_nodes(self):
    """
    函数名：ExtendedMetricTests.test_qini_keeps_defined_control_only_zero_nodes
    功能：验证传统 Qini 保留仅包含 Control 的已定义零值前缀节点。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        # Qini 曲线点为 (0,0)、(.25,0)、(.5,.25)、(.75,.25)、(1,.5)。
        # 曲线面积 3/16 减去随机基线面积 1/4，得到 -1/16。
        m = ranking_metrics([0, 1, 0, 1], [0, 1, 0, 1], [4, 3, 2, 1], 0.5)
        self.assertEqual(m['qini_score_traditional'], -1 / 16)

        # 连续两个对照组前缀的零值点都参与 v2 积分。
        y, t, scores = [0, 0, 1, 1], [0, 0, 1, 1], [4, 3, 2, 1]
        curve = uplift_curve(y, t, scores, 0.5)
        self.assertTrue(np.isnan(curve.qini_traditional.iloc[0]))
        np.testing.assert_array_equal(curve.qini_traditional.iloc[1:3], [0, 0])
        np.testing.assert_allclose(curve.qini_traditional.iloc[3:], [1 / 4, 1 / 2])

        def centered_area(mask):
    """
    函数名：ExtendedMetricTests.test_qini_keeps_defined_control_only_zero_nodes.centered_area
    功能：计算给定曲线节点相对端点随机基线的中心化面积。
    输入参数：
        - mask：用于选择参与积分的曲线节点的布尔掩码。
    输出参数：
        - 中心化曲线面积的浮点数。
    """
            x = np.r_[0.0, curve.loc[mask, 'fraction'].to_numpy()]
            q = np.r_[0.0, curve.loc[mask, 'qini_traditional'].to_numpy()]
            return np.sum(np.diff(x) * (q[1:] + q[:-1]) / 2) - q[-1] / 2

        v2 = centered_area(curve.n_control > 0)
        v1 = centered_area((curve.n_control > 0) & (curve.n_treatment > 0))
        self.assertEqual(v2, -1 / 8)
        self.assertEqual(v1, -1 / 16)
        self.assertEqual(ranking_metrics(y, t, scores, 0.5)['qini_score_traditional'], v2)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_relative_lift_undefined_for_nonpositive_total_effect —— 确认总体效果非正时相对提升保持未定义。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认总体效果非正时相对提升保持未定义。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_relative_lift_undefined_for_nonpositive_total_effect(self):
    """
    函数名：ExtendedMetricTests.test_relative_lift_undefined_for_nonpositive_total_effect
    功能：验证总体处理效应非正时，相对 lift 被标记为未定义而不产生误导数值。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        for y in [np.zeros(8), 1 - self.y]:
            c = uplift_curve(y, self.t, self.s, 0.5)
            self.assertTrue(c.relative_lift.isna().all())
            self.assertTrue(np.isfinite(c.gain_plugin.iloc[-1]))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_paired_ranking_intervals_same_score_zero_difference —— 确认相同排序的配对指标区间为零。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认相同排序的配对指标区间为零。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_paired_ranking_intervals_same_score_zero_difference(self):
    """
    函数名：ExtendedMetricTests.test_paired_ranking_intervals_same_score_zero_difference
    功能：验证相同策略评分的配对排序指标差值及其 Bootstrap 区间为零。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        r = ranking_confidence_intervals(
            self.y, self.t, {'propensity': self.s, 'same': self.s}, 0.5, 30
        )
        same = r[r.policy == 'same']
        np.testing.assert_array_equal(same.vs_propensity, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_low, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_high, 0)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_ranking_bootstrap_matches_independent_expanded_sample —— 用独立展开的重抽样记录核对排序区间。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：用独立展开的重抽样记录核对排序区间。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_ranking_bootstrap_matches_independent_expanded_sample(self):
    """
    函数名：ExtendedMetricTests.test_ranking_bootstrap_matches_independent_expanded_sample
    功能：验证排名指标 Bootstrap 与独立展开重抽样样本的计算结果一致。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        seed, draws = 19, 40
        score_dict = {'propensity': self.s, 'reverse': -self.s}
        r = ranking_confidence_intervals(self.y, self.t, score_dict, 0.5, draws, seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 271828]))
        a1, a0 = np.flatnonzero(self.t == 1), np.flatnonzero(self.t == 0)
        expected = []
        for _ in range(draws):
            ix = np.r_[rng.choice(a1, len(a1)), rng.choice(a0, len(a0))]
            # 重复记录的结果相同，因此这些记录之间的同分顺序不影响结果。
            rev = ranking_metrics(self.y[ix], self.t[ix], -self.s[ix], 0.5)[
                'qini_score_traditional'
            ]
            ref = ranking_metrics(self.y[ix], self.t[ix], self.s[ix], 0.5)[
                'qini_score_traditional'
            ]
            expected.append(rev - ref)
        actual = r[
            (r.policy == 'reverse') & (r.metric == 'qini_score_traditional')
        ].iloc[0]
        np.testing.assert_allclose(
            [actual.vs_propensity_ci_low, actual.vs_propensity_ci_high],
            np.quantile(expected, [0.025, 0.975]),
        )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_no_draws_and_invalid_bootstrap —— 核对关闭重抽样及非法重抽样次数的行为。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对关闭重抽样及非法重抽样次数的行为。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_no_draws_and_invalid_bootstrap(self):
    """
    函数名：ExtendedMetricTests.test_no_draws_and_invalid_bootstrap
    功能：验证关闭 Bootstrap 和传入非法重抽样次数时的边界行为。
    输入参数：
        - 无显式业务参数（实例方法中的 self 表示当前对象）。
    输出参数：
        - 无；通过断言验证预期行为，失败时由测试框架报告。
    """
        r = ranking_confidence_intervals(self.y, self.t, {'s': self.s}, 0.5, 0)
        self.assertTrue(r.ci_low.isna().all())
        self.assertTrue(r.vs_propensity.isna().all())
        for draws in [-1, 1, 2.5]:
            with self.assertRaises(ValueError):
                ranking_confidence_intervals(self.y, self.t, {'s': self.s}, 0.5, draws)
        with self.assertRaises(ValueError):
            ranking_confidence_intervals(self.y, self.t, {}, 0.5)


if __name__ == "__main__":
    unittest.main()
