# 项目测试：验证SQL 审计。

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from criteo_uplift import pipeline_io
from criteo_uplift.sql_audit import SCHEMA_SQL, audit_arrays, run_sql_audit


#-----------------------------------------------------------------------------------------------------------------------
# 类：SqlAuditTests —— 验证SQL 审计。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证SQL 审计。
#-----------------------------------------------------------------------------------------------------------------------
class SqlAuditTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
        # 让 CSV 原始行号、分数排序和数组位置刻意不同。
        self.ids = np.array([80, 3, 9, 12, 6, 200, 30, 2])
        self.y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        self.t = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        self.scores = np.arange(8, 0, -1)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：audit —— 使用测试预设数组执行 SQL 核对。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     **kwargs：传递给被调用函数的额外命名参数。
    # 输出：
    #     result：SQL 审计的结果表字典。
    # 说明：使用测试预设数组执行 SQL 核对。
    #-----------------------------------------------------------------------------------------------------------------------
    def audit(self, **kwargs):
        arguments = dict(
            row_id=self.ids,
            y=self.y,
            t=self.t,
            score_dict={"propensity": self.scores},
            e=0.5,
            budget=0.25,
            n_bins=4,
        )
        arguments.update(kwargs)
        return audit_arrays(**arguments)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_hand_calculated_cvr_bins_and_policy_value —— 用手算示例核对 SQL 转化率、分箱及策略收益。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：用手算示例核对 SQL 转化率、分箱及策略收益。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_hand_calculated_cvr_bins_and_policy_value(self):
        result = self.audit()
        cvr = result["cvr"].iloc[0]
        self.assertEqual(cvr.n_treatment, 4)
        self.assertEqual(cvr.n_control, 4)
        self.assertEqual(cvr.cvr_treatment, 0.75)
        self.assertEqual(cvr.cvr_control, 0.25)
        self.assertEqual(cvr.difference_in_means, 0.5)
        self.assertEqual(cvr.ate_ht, 0.5)
        rows = result["policy_values"].set_index("policy")
        # 前两条记录包含一个处理组正例，收益为 2/8 × 100000 = 25000。
        self.assertEqual(rows.loc["propensity", "sql_incremental_per_100k"], 25_000)
        self.assertEqual(rows.loc["random", "sql_incremental_per_100k"], 12_500)
        np.testing.assert_array_equal(rows.sql_minus_python, 0)
        bins = result["score_bins"]
        np.testing.assert_array_equal(bins.n, [2, 2, 2, 2])
        np.testing.assert_array_equal(
            bins.incremental_per_100k_eligible, [25_000, -25_000, 50_000, 0]
        )
        self.assertTrue(bins.observed_uplift.isna().all())
        self.assertTrue((result["join_integrity"].joined_rows == 8).all())

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_assignment_probability_is_fixed_and_not_test_treatment_fraction —— 确认 SQL 收益使用外部固定分配概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 SQL 收益使用外部固定分配概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_assignment_probability_is_fixed_and_not_test_treatment_fraction(self):
        result = self.audit(e=0.8)
        cvr = result["cvr"].iloc[0]
        self.assertEqual(cvr.difference_in_means, 0.5)
        self.assertAlmostEqual(cvr.ate_ht, (3 / 0.8 - 1 / 0.2) / 8)
        values = result["policy_values"].set_index("policy")
        self.assertEqual(values.loc["propensity", "sql_incremental_per_100k"], 15_625)
        np.testing.assert_allclose(values.sql_minus_python, 0, atol=1e-9)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_tied_scores_use_independent_priorities_shared_with_python —— 确认 SQL 与 Python 共用独立的同分优先级。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 SQL 与 Python 共用独立的同分优先级。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_tied_scores_use_independent_priorities_shared_with_python(self):
        n = 101
        ids = np.arange(n)[::-1]
        y = np.ones(n)
        t = np.r_[np.ones(50), np.zeros(51)]
        result = audit_arrays(
            ids,
            y,
            t,
            {"constant": np.ones(n), "copy": np.ones(n)},
            e=0.5,
            budget=0.2,
            seed=17,
            n_bins=10,
        )
        priorities = np.random.default_rng(17).random(n)
        chosen = np.argsort(priorities)[:20]
        hand = (2 * t[chosen] - 2 * (1 - t[chosen])).sum() / n * 100_000
        values = result["policy_values"].set_index("policy")
        self.assertAlmostEqual(values.loc["constant", "sql_incremental_per_100k"], hand)
        self.assertEqual(
            values.loc["copy", "sql_incremental_per_100k"],
            values.loc["constant", "sql_incremental_per_100k"],
        )
        np.testing.assert_allclose(values.sql_minus_python, 0, atol=1e-9)
        bins = result["score_bins"].query("policy == 'constant'")
        self.assertEqual(bins.n.sum(), 101)
        self.assertEqual(bins.n.iloc[0], 11)
        self.assertTrue((bins.n.iloc[1:] == 10).all())

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_zero_and_full_budget_have_correct_denominators —— 核对零预算及全量预算的收益分母。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对零预算及全量预算的收益分母。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_zero_and_full_budget_have_correct_denominators(self):
        zero = self.audit(budget=0)["policy_values"]
        np.testing.assert_array_equal(zero.n_selected, 0)
        np.testing.assert_array_equal(zero.sql_incremental_per_100k, 0)
        full = self.audit(budget=1)["policy_values"]
        np.testing.assert_array_equal(full.n_selected, 8)
        np.testing.assert_array_equal(full.sql_incremental_per_100k, 50_000)
        np.testing.assert_array_equal(full.vs_random_per_100k, 0)
        rounded = self.audit(budget=0.3)["policy_values"]
        np.testing.assert_array_equal(rounded.selected_fraction, 0.25)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_schema_rejects_duplicate_and_orphan_keys —— 确认 SQL 表约束拒绝重复键及无对应记录的键。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 SQL 表约束拒绝重复键及无对应记录的键。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_schema_rejects_duplicate_and_orphan_keys(self):
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute("INSERT INTO assignment VALUES (10, 0, 1)")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO assignment VALUES (10, 1, 0)")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO outcome VALUES (999, 1)")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO policy_scores VALUES ('test', 999, 0.1, 0.5)")

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_inputs_fail_before_sql —— 确认非法输入在执行 SQL 前被拒绝。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认非法输入在执行 SQL 前被拒绝。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_inputs_fail_before_sql(self):
        for kwargs in (
            {"row_id": np.zeros(8, dtype=int)},
            {"row_id": self.ids.astype(float)},
            {"row_id": self.ids[:-1]},
            {"row_id": -self.ids},
            {"n_bins": 0},
            {"score_dict": {}},
            {"score_dict": {"bad": np.full(8, np.nan)}},
            {"e": 1},
            {"budget": -0.1},
            {"t": np.ones(8)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.audit(**kwargs)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_runner_preserves_source_and_rejects_overwrites_and_data —— 确认 SQL 审计保留来源并拒绝覆盖及写入数据目录。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 SQL 审计保留来源并拒绝覆盖及写入数据目录。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_runner_preserves_source_and_rejects_overwrites_and_data(self):
        with tempfile.TemporaryDirectory(prefix="sql_audit_test_") as tmp:
            root = Path(tmp)
            source = root / "frozen"
            source.mkdir()
            (source / "models").mkdir()
            (source / "evaluation").mkdir()
            np.savez(
                source / "evaluation" / "predictions.npz",
                row_id=self.ids,
                y=self.y,
                t=self.t,
                propensity=self.scores,
            )
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "assignment_probability": 0.5,
                        "seed": 42,
                        "primary_outcome": "conversion",
                    }
                )
            )
            (source / "models" / "selection.json").write_text(
                json.dumps(
                    {
                        "frozen_before_test": True,
                        "primary_budget": 0.25,
                        "selected_policy": "propensity",
                    }
                )
            )
            before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            output = root / "audit"
            result = run_sql_audit(source, output)
            self.assertEqual(result["n_eligible"], 8)
            self.assertTrue(result["source_unchanged"])
            self.assertEqual(result["maximum_absolute_sql_python_difference"], 0)
            self.assertIn("ROW_NUMBER()", (output / "queries.sql").read_text())
            self.assertEqual({str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}, before)
            with self.assertRaises(FileExistsError):
                run_sql_audit(source, output)
            with self.assertRaises(ValueError):
                run_sql_audit(source, source / "new_audit")
            protected = root / "data"
            protected.mkdir()
            alias = root / "link"
            alias.symlink_to(protected, target_is_directory=True)
            with patch.object(pipeline_io, "DATA_ROOT", protected):
                with self.assertRaises(ValueError):
                    run_sql_audit(source, alias / "audit")
            self.assertEqual(list(protected.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
