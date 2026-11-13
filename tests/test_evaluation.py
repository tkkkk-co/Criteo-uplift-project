# 项目测试：验证效果估计和策略评价。

import math
from io import StringIO
import pandas as pd
import unittest

import numpy as np

from criteo_uplift.metrics import RANKING_METRICS, evaluate_ranking
from criteo_uplift.evaluation import _normal_pvalue, ate_summary_from_audit, ate_summary


#-----------------------------------------------------------------------------------------------------------------------
# 类：EvaluationTests —— 验证效果估计和策略评价。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证效果估计和策略评价。
#-----------------------------------------------------------------------------------------------------------------------
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        #-------------
        # 函数名：EvaluationTests.setUp
        # 功能：准备 ATE、HT 曲线和 Policy Evaluation 测试共用的二元样本及策略分数。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化或清理测试实例状态。
        #-------------
        self.y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        self.t = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        self.scores = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)

    def test_difference_in_means_and_pooled_null(self):
        #-------------
        # 函数名：EvaluationTests.test_difference_in_means_and_pooled_null
        # 功能：验证 ATE 差均值估计、标准误和 pooled 零假设检验与手工公式一致。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        result = ate_summary(self.y, self.t)
        self.assertEqual(result["ate"], 0.5)
        self.assertEqual(result["relative_uplift"], 2.0)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(0.125))
        self.assertAlmostEqual(result["p_value"], math.erfc(1.0))
        self.assertLess(result["ci_low"], result["ate"])
        self.assertGreater(result["ci_high"], result["ate"])

    def test_ate_row_and_audit_paths_agree_for_every_statistic(self):
        #-------------
        # 函数名：EvaluationTests.test_ate_row_and_audit_paths_agree_for_every_statistic
        # 功能：验证逐行 ATE 与 Audit 计数 ATE 的估计、标准误、置信区间和 p 值字段一致。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_normal_tail_retains_subnormal_probability_and_log_beyond_float_range(self):
        #-------------
        # 函数名：EvaluationTests.test_normal_tail_retains_subnormal_probability_and_log_beyond_float_range
        # 功能：验证极端正态尾部概率在浮点下溢附近仍保留可用的 log p 值和文本表示。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
        #-------------
        # 函数名：EvaluationTests.test_ate_extreme_pvalues_preserve_test_and_estimates
        # 功能：验证极端 p 值场景下检验结果不会改变 ATE 点估计及其区间。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
        #-------------
        # 函数名：EvaluationTests.test_aggregate_ate_rejects_invalid_counts
        # 功能：验证 ATE 计数聚合函数拒绝非法人数、正例数和非整数计数。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for n, positive in ((0, 0), (2, 3), (2, -1), (2.5, 1), (True, 1)):
            audit = {'groups': {
                '0': {'n': n, 'conversion': {'positive': positive}},
                '1': {'n': 3, 'conversion': {'positive': 1}},
            }}
            with self.subTest(n=n, positive=positive), self.assertRaises(ValueError):
                ate_summary_from_audit(audit, 'conversion')

    def test_constant_outcomes_and_missing_control_response(self):
        #-------------
        # 函数名：EvaluationTests.test_constant_outcomes_and_missing_control_response
        # 功能：验证常量结果和某一处理臂缺失正例时 ATE 的边界计算行为。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        result = ate_summary(np.zeros(8), self.t)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["standard_error"], 0.0)
        self.assertTrue(np.isnan(result["relative_uplift"]))

    def test_ht_curve_preserves_fixed_assignment_probability(self):
        #-------------
        # 函数名：EvaluationTests.test_ht_curve_preserves_fixed_assignment_probability
        # 功能：验证 HT 曲线始终使用调用方传入的固定处理分配概率。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        curve = evaluate_ranking(self.y, self.t, {"model": self.scores}, e=0.8)["curves"]
        expected_z = self.t * self.y / 0.8 - (1 - self.t) * self.y / 0.2
        np.testing.assert_allclose(curve.gain_ht, np.r_[0.0, expected_z.cumsum()] / 8)
        self.assertNotAlmostEqual(
            curve.gain_ht.iloc[-1], ate_summary(self.y, self.t)["ate"]
        )
        self.assertTrue(np.isnan(curve.qini_traditional.iloc[2]))
        self.assertAlmostEqual(curve.qini_traditional.iloc[3], -1 / 8)
        self.assertAlmostEqual(curve.uplift_rate.iloc[3], -0.5)
        self.assertTrue(np.isnan(curve.uplift_rate.iloc[2]))

    def test_area_uses_full_curve_and_centering(self):
        #-------------
        # 函数名：EvaluationTests.test_area_uses_full_curve_and_centering
        # 功能：验证曲线积分使用完整前缀分辨率，并按终点随机基线进行中心化。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        result = evaluate_ranking(self.y, self.t, {"model": self.scores}, e=0.5)
        curve, metrics = result["curves"], result["metrics"]
        metrics_only = evaluate_ranking(
            self.y, self.t, {"model": self.scores}, e=0.5, include_curves=False
        )
        self.assertIsNone(metrics_only["curves"])
        pd.testing.assert_frame_equal(metrics_only["metrics"], metrics)
        np.testing.assert_array_equal(curve.n_selected, np.arange(9))
        points = curve.dropna(subset=["gain_plugin"])
        self.assertAlmostEqual(metrics.auuc.iloc[0], np.trapezoid(points.gain_plugin, points.fraction))
        self.assertEqual(set(metrics.columns) & set(RANKING_METRICS), set(RANKING_METRICS))
        self.assertEqual(len(metrics.columns), 9)

    def test_traditional_qini_bridges_initial_undefined_prefix(self):
        #-------------
        # 函数名：EvaluationTests.test_traditional_qini_bridges_initial_undefined_prefix
        # 功能：验证传统 Qini 在初始前缀未同时出现两组时采用规定的积分连接方式。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        y, t = [1, 0, 0, 1], [1, 1, 0, 0]
        scores = [4, 3, 2, 1]
        result = evaluate_ranking(y, t, {"model": scores}, e=0.5)
        curve, metrics = result["curves"], result["metrics"]
        # 传统 Qini 的有效点为 (3/4, 1/4)、(1, 0)，并加入原点。
        # 两段梯形面积分别为 3/32 和 1/32，随机基线面积为零。
        self.assertTrue(curve.qini_traditional.iloc[:3].isna().all())
        self.assertEqual(curve.qini_traditional.iloc[-1], 0)
        self.assertEqual(metrics.qini_score.iloc[0], 1 / 8)

    def test_traditional_qini_centers_its_own_full_coverage_endpoint(self):
        #-------------
        # 函数名：EvaluationTests.test_traditional_qini_centers_its_own_full_coverage_endpoint
        # 功能：验证传统 Qini 使用自身全覆盖终点计算随机基线面积。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        y, t = [1, 0, 0, 0], [1, 1, 0, 0]
        scores = [4, 3, 2, 1]
        result = evaluate_ranking(y, t, {"model": scores}, e=0.6)
        curve, metrics = result["curves"], result["metrics"]
        # 传统 Qini 终点为 1/4，原始面积为 3/32 + 1/16 = 5/32。
        # 扣除随机基线的三角形面积 1/8，得到 1/32。
        self.assertEqual(curve.qini_traditional.iloc[-1], 1 / 4)
        self.assertEqual(metrics.qini_score.iloc[0], 1 / 32)

    def test_traditional_score_is_not_changed_by_ht_assignment_probability(self):
        #-------------
        # 函数名：EvaluationTests.test_traditional_score_is_not_changed_by_ht_assignment_probability
        # 功能：验证传统 Qini 排序分数不受 HT 处理分配概率参数影响。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        a = evaluate_ranking(self.y, self.t, {"model": self.scores}, e=0.5, include_curves=False)["metrics"]
        b = evaluate_ranking(self.y, self.t, {"model": self.scores}, e=0.8, include_curves=False)["metrics"]
        self.assertEqual(a.qini_score.iloc[0], b.qini_score.iloc[0])

    def test_tie_break_is_seeded_and_does_not_use_row_order(self):
        #-------------
        # 函数名：EvaluationTests.test_tie_break_is_seeded_and_does_not_use_row_order
        # 功能：验证相同分数的排序由独立随机种子决定，而不依赖输入行顺序。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        n = 100
        t = np.r_[np.ones(n // 2), np.zeros(n // 2)]
        y = np.ones(n)
        a = evaluate_ranking(y, t, {"model": np.ones(n)}, e=0.5, seed=42)["curves"]
        b = evaluate_ranking(y, t, {"model": np.ones(n)}, e=0.5, seed=42)["curves"]
        c = evaluate_ranking(y, t, {"model": np.ones(n)}, e=0.5, seed=43)["curves"]
        np.testing.assert_array_equal(a.gain_ht, b.gain_ht)
        self.assertFalse(np.array_equal(a.gain_ht, c.gain_ht))
        self.assertGreater(a.n_control.iloc[10], 0)

    def test_weighted_prefix_reselects_boundary(self):
        #-------------
        # 函数名：EvaluationTests.test_weighted_prefix_reselects_boundary
        # 功能：验证 Bootstrap 权重压缩表示在 top-k 边界处正确重新选择部分样本。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        # 展开重复记录后，直接核对每个预算的人数和累计 HT Gain。
        counts = np.array([0, 2, 3, 1])
        y = np.repeat([0, 1, 1, 0], counts)
        t = np.repeat([0, 1, 0, 1], counts)
        scores = np.repeat([4, 3, 2, 1], counts)
        k = np.array([0, 1, 2, 3, 5, 6])
        result = evaluate_ranking(y, t, {"model": scores}, .5, budgets=k / 6, primary_budget=.5)
        expected = np.r_[0, np.cumsum(2 * t * y - 2 * (1 - t) * y)][k] / 6
        np.testing.assert_allclose(result["metrics"].gain_ht, expected)

    def test_same_score_paired_difference_is_exactly_zero(self):
        #-------------
        # 函数名：EvaluationTests.test_same_score_paired_difference_is_exactly_zero
        # 功能：验证两个完全相同的策略分数在配对比较中的增量差值严格为零。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        result = evaluate_ranking(
            self.y, self.t, {"propensity": self.scores, "T": self.scores},
            e=.5, budgets=[.25, .5, 1.], n_bootstrap=40, seed=19,
        )
        pairs = result["bootstrap"]["pairwise"]
        same = pairs.query("policy == 'T' and reference == 'propensity'")
        np.testing.assert_array_equal(same[["difference", "ci_low", "ci_high"]], 0)
        full = pairs.query("budget == 1 and reference == 'random'")
        np.testing.assert_array_equal(full[["difference", "ci_low", "ci_high"]], 0)
        wins = result["bootstrap"]["selection_frequency"]
        self.assertNotIn("random", set(wins.policy))
        self.assertTrue(wins.query("policy == 'propensity'").win_frequency.eq(1).all())

    def test_bootstrap_is_reproducible_and_point_estimates_unchanged(self):
        #-------------
        # 函数名：EvaluationTests.test_bootstrap_is_reproducible_and_point_estimates_unchanged
        # 功能：验证 Policy Bootstrap 在相同随机种子下可复现，且不改变原始点估计。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        kwargs = dict(
            y=self.y,
            t=self.t,
            score_dict={"S": self.scores},
            e=0.5,
            budgets=[0.5],
            seed=77,
        )
        a = evaluate_ranking(**kwargs, n_bootstrap=30)
        b = evaluate_ranking(**kwargs, n_bootstrap=30)
        point = evaluate_ranking(**kwargs, n_bootstrap=0)
        for name in ("summary", "pairwise", "selection_frequency"):
            pd.testing.assert_frame_equal(a["bootstrap"][name], b["bootstrap"][name], check_exact=True)
        pd.testing.assert_frame_equal(a["metrics"], point["metrics"])
        self.assertIsNone(point["bootstrap"])

    def test_budget_rounding_and_analytic_random(self):
        #-------------
        # 函数名：EvaluationTests.test_budget_rounding_and_analytic_random
        # 功能：验证预算比例向下取整得到 top-k 人数，并正确计算随机策略解析期望。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        result = evaluate_ranking(self.y, self.t, {"S": self.scores}, .5,
                                  budgets=[.01, .3], primary_budget=.3, n_bootstrap=2)
        np.testing.assert_array_equal(result["metrics"].n_selected, [0, 2])
        random = result["bootstrap"]["summary"].query("policy == 'random'")
        np.testing.assert_array_equal(random.point_estimate, [0, .125])

    def test_paired_difference_matches_direct_resampling(self):
        #-------------
        # 函数名：EvaluationTests.test_paired_difference_matches_direct_resampling
        # 功能：验证优化后的配对 Bootstrap 差值区间与显式直接重抽样结果一致。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        # 通过显式展开抽样行，独立核对优化后的重抽样计算。
        budgets = np.array([0.25, 0.3, 0.5])
        seed, replicates = 15, 30
        result = evaluate_ranking(self.y, self.t, {"S": self.scores}, .5,
                                  budgets=budgets, primary_budget=.25, n_bootstrap=replicates, seed=seed)
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
            delta.append((gain - np.floor(budgets * 8) / 8 * z[sample].mean()) * 100_000)
        expected = np.quantile(delta, [0.025, 0.975], axis=0)
        selected = result["bootstrap"]["pairwise"].query("policy == 'S' and reference == 'random'")
        np.testing.assert_allclose(selected.ci_low * 100_000, expected[0])
        np.testing.assert_allclose(selected.ci_high * 100_000, expected[1])

    def test_thirty_percent_preserves_existing_budgets_and_paired_intervals(self):
        #-------------
        # 函数名：EvaluationTests.test_thirty_percent_preserves_existing_budgets_and_paired_intervals
        # 功能：验证新增 30% 不改变已有预算的估计和配对区间，并核对人数、边际贡献与全量终点。
        # 输入参数：
        #     - 无显式业务参数（self 为当前测试实例）。
        # 输出参数：
        #     - 无；通过断言验证预算扩展不改变原统计口径。
        #-------------
        scores = {"S": self.scores, "propensity": self.scores[::-1]}
        old_budgets = [.02, .05, .1, .2, .5, 1.]
        old = evaluate_ranking(self.y, self.t, scores, .5, budgets=old_budgets, n_bootstrap=30)
        expanded = evaluate_ranking(self.y, self.t, scores, .5, budgets=[*old_budgets, .3], n_bootstrap=30)
        for name in ("summary", "pairwise", "selection_frequency"):
            retained = expanded["bootstrap"][name].query("budget != .3").reset_index(drop=True)
            pd.testing.assert_frame_equal(old["bootstrap"][name], retained, check_exact=True)
        points = expanded["metrics"]
        self.assertEqual(set(points.budget), {*old_budgets, .3})
        np.testing.assert_array_equal(points.n_selected, np.floor(points.budget * len(self.y)))
        self.assertEqual(points.query("budget == 1").gain_ht.nunique(), 1)

    def test_invalid_inputs_fail_explicitly(self):
        #-------------
        # 函数名：EvaluationTests.test_invalid_inputs_fail_explicitly
        # 功能：验证 ATE 与排序评价函数对非法输入明确报错。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with self.assertRaises(ValueError):
            ate_summary([1, 0], [1, 1])
        with self.assertRaises(ValueError):
            ate_summary([1, np.nan], [1, 0])
        with self.assertRaises(ValueError):
            evaluate_ranking(self.y, self.t, {"model": self.scores}, e=1)
        with self.assertRaises(ValueError):
            evaluate_ranking(self.y, self.t, {"model": [1, 2]}, e=0.5, include_curves=False)
        with self.assertRaises(ValueError):
            evaluate_ranking(self.y, self.t, {"random": self.scores}, .5, n_bootstrap=0)
        with self.assertRaises(ValueError):
            evaluate_ranking(self.y, self.t, {"s": self.scores}, .5, budgets=[1.2], n_bootstrap=0)


#-----------------------------------------------------------------------------------------------------------------------
# 类：ExtendedMetricTests —— 验证效果估计和策略评价。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证效果估计和策略评价。
#-----------------------------------------------------------------------------------------------------------------------
class ExtendedMetricTests(unittest.TestCase):
    def setUp(self):
        #-------------
        # 函数名：ExtendedMetricTests.setUp
        # 功能：准备扩展排序指标测试使用的八行手工数据和评分向量。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化或清理测试实例状态。
        #-------------
        self.t = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        self.s = np.arange(8, 0, -1)

    def test_full_eight_row_hand_example(self):
        #-------------
        # 函数名：ExtendedMetricTests.test_full_eight_row_hand_example
        # 功能：使用八行手工样本核对 uplift curve、HT gain 和 Qini 指标的完整计算结果。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        result = evaluate_ranking(self.y, self.t, {"model": self.s}, 0.5)
        c = result["curves"]
        np.testing.assert_allclose(
            c.uplift_rate.iloc[2:], [1, 1, 1, 2 / 3, 1 / 3, 5 / 12, 1 / 2]
        )
        np.testing.assert_allclose(
            c.gain_plugin.iloc[2:], [1 / 4, 3 / 8, 1 / 2, 5 / 12, 1 / 4, 35 / 96, 1 / 2]
        )
        np.testing.assert_allclose(
            c.qini_traditional.iloc[2:],
            [1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8, 5 / 24, 1 / 4],
        )
        m = result["metrics"].iloc[0]
        self.assertAlmostEqual(m.auuc, 81 / 256)
        self.assertAlmostEqual(m.qini_score, 19 / 384)

    def test_qini_keeps_defined_control_only_zero_nodes(self):
        #-------------
        # 函数名：ExtendedMetricTests.test_qini_keeps_defined_control_only_zero_nodes
        # 功能：验证传统 Qini 保留仅包含 Control 的已定义零值前缀节点。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        # Qini 曲线点为 (0,0)、(.25,0)、(.5,.25)、(.75,.25)、(1,.5)。
        # 曲线面积 3/16 减去随机基线面积 1/4，得到 -1/16。
        m = evaluate_ranking([0, 1, 0, 1], [0, 1, 0, 1], {"model": [4, 3, 2, 1]}, 0.5, include_curves=False)["metrics"]
        self.assertEqual(m.qini_score.iloc[0], -1 / 16)

        # 连续两个对照组前缀的零值点都参与积分。
        y, t, scores = [0, 0, 1, 1], [0, 0, 1, 1], [4, 3, 2, 1]
        result = evaluate_ranking(y, t, {"model": scores}, 0.5)
        curve = result["curves"]
        self.assertTrue(np.isnan(curve.qini_traditional.iloc[0]))
        np.testing.assert_array_equal(curve.qini_traditional.iloc[1:3], [0, 0])
        np.testing.assert_allclose(curve.qini_traditional.iloc[3:], [1 / 4, 1 / 2])

        def centered_area(mask):
            #-------------
            # 函数名：ExtendedMetricTests.test_qini_keeps_defined_control_only_zero_nodes.centered_area
            # 功能：计算给定曲线节点相对端点随机基线的中心化面积。
            # 输入参数：
            #     - mask：用于选择参与积分的曲线节点的布尔掩码。
            # 输出参数：
            #     - 中心化曲线面积的浮点数。
            #-------------
            x = np.r_[0.0, curve.loc[mask, 'fraction'].to_numpy()]
            q = np.r_[0.0, curve.loc[mask, 'qini_traditional'].to_numpy()]
            return np.sum(np.diff(x) * (q[1:] + q[:-1]) / 2) - q[-1] / 2

        with_control_only = centered_area(curve.n_control > 0)
        with_both_arms = centered_area((curve.n_control > 0) & (curve.n_treatment > 0))
        self.assertEqual(with_control_only, -1 / 8)
        self.assertEqual(with_both_arms, -1 / 16)
        self.assertEqual(result["metrics"].qini_score.iloc[0], with_control_only)

    def test_gain_remains_defined_for_nonpositive_total_effect(self):
        #-------------
        # 函数名：ExtendedMetricTests.test_gain_remains_defined_for_nonpositive_total_effect
        # 功能：验证总体处理效应非正时，绝对转化率差和 Uplift Gain 仍按定义计算。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for y in [np.zeros(8), 1 - self.y]:
            c = evaluate_ranking(y, self.t, {"model": self.s}, 0.5)["curves"]
            expected = y[self.t == 1].mean() - y[self.t == 0].mean()
            self.assertLessEqual(expected, 0)
            self.assertEqual(c.uplift_rate.iloc[-1], expected)
            self.assertEqual(c.gain_plugin.iloc[-1], expected)

    def test_paired_ranking_intervals_same_score_zero_difference(self):
        #-------------
        # 函数名：ExtendedMetricTests.test_paired_ranking_intervals_same_score_zero_difference
        # 功能：验证相同策略评分的配对排序指标差值及其 Bootstrap 区间为零。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        r = evaluate_ranking(self.y, self.t, {"propensity": self.s, "same": self.s},
                             .5, primary_metric="qini_score", n_bootstrap=30)
        same = r["bootstrap"]["pairwise"].query("policy == 'same' and reference == 'propensity'")
        np.testing.assert_array_equal(same[["difference", "ci_low", "ci_high"]], 0)

    def test_primary_bootstrap_matches_independent_expanded_sample(self):
        #-------------
        # 功能：逐个检查五种可选主指标的均值、有效重复与配对区间。
        # 输入：八行样本和独立重建的分层抽样记录。
        # 输出：验证统一入口每次只输出指定主指标的推断。
        #-------------
        score_dict = {"propensity": self.s, "reverse": -self.s}
        seed, draws, fraction = 19, 40, .25
        arms = [np.flatnonzero(self.t == arm) for arm in (1, 0)]
        for metric in RANKING_METRICS:
            with self.subTest(metric=metric):
                result = evaluate_ranking(self.y, self.t, score_dict, .5, budgets=[fraction],
                                          primary_metric=metric, primary_budget=fraction,
                                          n_bootstrap=draws, seed=seed, include_curves=False)
                rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
                values = []
                for _ in range(draws):
                    ix = np.r_[rng.choice(arms[0], len(arms[0])), rng.choice(arms[1], len(arms[1]))]
                    point = evaluate_ranking(self.y[ix], self.t[ix], {k: s[ix] for k, s in score_dict.items()},
                                             .5, budgets=[fraction], primary_metric=metric,
                                             primary_budget=fraction, n_bootstrap=0, include_curves=False)
                    values.append(point["metrics"][metric].to_numpy())
                values = np.asarray(values)
                inference = result["bootstrap"]
                self.assertEqual(set(inference["summary"].metric), {metric})
                candidates = inference["summary"].query("policy != 'random'")
                for j, row in enumerate(candidates.itertuples()):
                    valid = values[np.isfinite(values[:, j]), j]
                    self.assertEqual(row.valid_replicates, len(valid))
                    self.assertAlmostEqual(row.bootstrap_mean, valid.mean())
                pair = inference["pairwise"].query("policy == 'reverse' and reference == 'propensity'").iloc[0]
                delta = values[:, 1] - values[:, 0]
                delta = delta[np.isfinite(delta)]
                self.assertEqual(pair.valid_replicates, len(delta))
                np.testing.assert_allclose([pair.ci_low, pair.ci_high], np.quantile(delta, [.025, .975]))

    def test_no_draws_and_invalid_bootstrap(self):
        #-------------
        # 功能：检查关闭 Bootstrap 和非法主指标、预算、抽样次数。
        #-------------
        args = (self.y, self.t, {"s": self.s}, .5)
        result = evaluate_ranking(*args, n_bootstrap=0, include_curves=False)
        self.assertIsNone(result["bootstrap"])
        self.assertIsNone(result["curves"])
        for invalid in ({"n_bootstrap": -1}, {"n_bootstrap": 1}, {"n_bootstrap": 2.5},
                        {"primary_metric": "unknown"}, {"primary_budget": 0}):
            with self.assertRaises(ValueError):
                evaluate_ranking(*args, **invalid)
        missing = evaluate_ranking(*args, budgets=[.125], primary_metric="gain_plugin",
                                   primary_budget=.125, n_bootstrap=2)
        row = missing["bootstrap"]["summary"].query("policy == 's'").iloc[0]
        self.assertEqual(row.valid_replicates, 0)
        self.assertTrue(np.isnan(row.ci_low))
        area = evaluate_ranking(*args, budgets=[.5], primary_metric="auuc",
                                primary_budget=None, n_bootstrap=2)
        self.assertEqual(area["metrics"].budget.tolist(), [.5])
        self.assertTrue(area["bootstrap"]["summary"].budget.isna().all())


if __name__ == "__main__":
    unittest.main()
