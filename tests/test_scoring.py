# 项目测试：验证冻结策略离线打分。

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from criteo_uplift import pipeline_io
from criteo_uplift.config import FEATURES
from criteo_uplift.pipeline_io import fingerprint
from criteo_uplift.metrics import _rank
from score_policy import score_policy


#-----------------------------------------------------------------------------------------------------------------------
# 类：_FeatureOnlyModel —— 检查输入特征和分块大小，并返回预设测试分数的模型。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：检查输入特征和分块大小，并返回预设测试分数的模型。
#-----------------------------------------------------------------------------------------------------------------------
class _FeatureOnlyModel:
    n_features_in_ = 12

    def __init__(self, constant=False, max_rows=7):
        #-------------
        # 函数名：_FeatureOnlyModel.__init__
        # 功能：设置测试模型的常数分数模式和单次 predict 最大行数。
        # 输入参数：
        #     - constant：是否让 predict 返回相同分数。
        #     - max_rows：单次预测允许的最大输入行数。
        # 输出参数：
        #     - 无；保存测试模型配置。
        #-------------
        self.constant = constant
        self.max_rows = max_rows

    def _check_features(self, X):
        #-------------
        # 函数名：_FeatureOnlyModel._check_features
        # 功能：检查输入 DataFrame 仅包含按顺序排列的 f0–f11，并且分块行数未超限。
        # 输入参数：
        #     - X：当前批次的特征 DataFrame。
        # 输出参数：
        #     - 无；不符合测试约束时抛出 AssertionError。
        #-------------
        if list(X.columns) != FEATURES or len(X) > self.max_rows:
            raise AssertionError(
                "Only ordered pre-treatment features in bounded chunks are allowed."
            )

    def predict(self, X):
        #-------------
        # 函数名：_FeatureOnlyModel.predict
        # 功能：按测试设置返回常数分数或第一列特征值作为 CATE 分数。
        # 输入参数：
        #     - X：当前批次的 f0–f11 特征 DataFrame。
        # 输出参数：
        #     - 与输入行顺序一致的策略分数数组。
        #-------------
        self._check_features(X)
        return np.ones(len(X)) if self.constant else X.f0.to_numpy()

    def predict_outcomes(self, X):
        #-------------
        # 函数名：_FeatureOnlyModel.predict_outcomes
        # 功能：返回两个刻意不同的潜在结果分数，用于核验 propensity 路径选择了 mu1。
        # 输入参数：
        #     - X：当前批次的 f0–f11 特征 DataFrame。
        # 输出参数：
        #     - 二元组：(测试用 mu0 分数数组, 测试用 mu1 分数数组)。
        #-------------
        self._check_features(X)
        # 构造相反排序，以区分是否使用了 mu1、增量分数或 mu0。
        score = X.f0.to_numpy() / 100.0
        return score * 2, score


#-----------------------------------------------------------------------------------------------------------------------
# 类：ScoringTests —— 验证冻结策略离线打分。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证冻结策略离线打分。
#-----------------------------------------------------------------------------------------------------------------------
class ScoringTests(unittest.TestCase):
    def setUp(self):
        #-------------
        # 函数名：ScoringTests.setUp
        # 功能：创建临时 run、模型文件和只含前置特征的新人群 CSV，供 Score Policy 测试复用。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化或清理测试实例状态。
        #-------------
        self.directory = tempfile.TemporaryDirectory(prefix="criteo_scoring_test_")
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.source = self.root / "users.csv"
        frame = pd.DataFrame(
            {feature: np.arange(23, dtype=float) for feature in FEATURES}
        )
        frame["treatment"] = "unused text"
        frame["exposure"] = "post-treatment value"
        frame["conversion"] = "outcome must not be used"
        frame.to_csv(self.source, index=False)
        self.output = self.root / "results" / "scores.csv"
        self._save_run("t_logistic")

    def tearDown(self):
        #-------------
        # 函数名：ScoringTests.tearDown
        # 功能：清理 Score Policy 测试使用的临时目录。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化或清理测试实例状态。
        #-------------
        self.directory.cleanup()

    def _save_run(self, policy, model=None):
        #-------------
        # 函数名：ScoringTests._save_run
        # 功能：构造包含冻结 selection、manifest 和可选测试模型的最小实验目录。
        # 输入参数：
        #     - policy：测试用的冻结策略名称。
        #     - model：可选的测试模型实例；未提供时不保存模型文件。
        # 输出参数：
        #     - 保存模型与冻结元数据后的实验目录 pathlib.Path。
        #-------------
        selection = {"selected_policy": policy, "frozen_before_test": True}
        manifest = {"features": FEATURES, "seed": 42, "selection": selection}
        (self.run / "models").mkdir(exist_ok=True)
        (self.run / "models" / "selection.json").write_text(json.dumps(selection))
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        if policy != "random":
            name = "t_xgboost" if policy == "propensity" else policy
            joblib.dump(model or _FeatureOnlyModel(), self.run / "models" / f"{name}.joblib")

    def _score(self, **kwargs):
        #-------------
        # 函数名：ScoringTests._score
        # 功能：使用测试输入和小分块大小调用 score_policy，并允许覆盖指定参数。
        # 输入参数：
        #     - **kwargs：传递给 score_policy 的额外关键字参数。
        # 输出参数：
        #     - score_policy 返回的打分摘要字典。
        #-------------
        return score_policy(self.run, self.source, self.output, chunk_size=7, **kwargs)

    def test_feature_only_chunks_exact_budget_and_input_unchanged(self):
        #-------------
        # 函数名：ScoringTests.test_feature_only_chunks_exact_budget_and_input_unchanged
        # 功能：验证 Score Policy 只读取前置特征、分块处理、精确选择预算人数且不修改输入文件。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        before = fingerprint(self.source)
        result = self._score(budget=0.2)
        output = pd.read_csv(self.output)
        self.assertEqual(list(output), ["source_row_id", "score", "selected"])
        np.testing.assert_array_equal(output.source_row_id, np.arange(23))
        self.assertEqual(result["selected_n"], 4)
        self.assertEqual(int(output.selected.sum()), 4)
        np.testing.assert_array_equal(
            output.loc[output.selected == 1, "source_row_id"], [19, 20, 21, 22]
        )
        self.assertEqual(fingerprint(self.source), before)

    def test_propensity_uses_treated_response_probability(self):
        #-------------
        # 函数名：ScoringTests.test_propensity_uses_treated_response_probability
        # 功能：验证 propensity 策略使用 P(Y=1|X,T=1)，而不是处理分配概率或 Control 潜在结果。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        self._save_run("propensity")
        self._score(budget=0.2)
        output = pd.read_csv(self.output)
        np.testing.assert_allclose(output.score, np.arange(23) / 100.0)
        np.testing.assert_array_equal(
            output.loc[output.selected == 1, "source_row_id"], [19, 20, 21, 22]
        )

    def test_random_scores_and_tie_breaking_are_chunk_independent(self):
        #-------------
        # 函数名：ScoringTests.test_random_scores_and_tie_breaking_are_chunk_independent
        # 功能：验证随机策略分数和同分打破方式不受 CSV 分块大小影响。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for policy, model in (
            ("random", None),
            ("t_logistic", _FeatureOnlyModel(constant=True)),
        ):
            with self.subTest(policy=policy):
                self._save_run(policy, model)
                path1 = self.root / f"{policy}_1.csv"
                path2 = self.root / f"{policy}_2.csv"
                score_policy(self.run, self.source, path1, budget=0.5, chunk_size=7)
                score_policy(self.run, self.source, path2, budget=0.5, chunk_size=4)
                first, second = pd.read_csv(path1), pd.read_csv(path2)
                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(int(first.selected.sum()), 11)
                self.assertNotEqual(
                    first.loc[first.selected == 1, "source_row_id"].tolist(),
                    list(range(11)),
                )

    def test_budget_boundaries_and_decimal_floor(self):
        #-------------
        # 函数名：ScoringTests.test_budget_boundaries_and_decimal_floor
        # 功能：验证 0%、100% 和小数预算边界按 Decimal 向下取整得到正确 Top-k 人数。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        self._save_run("random")
        frame = pd.DataFrame({feature: np.arange(100) for feature in FEATURES})
        frame.to_csv(self.source, index=False)
        for budget, expected in ((0, 0), (1, 100), (0.58, 58)):
            path = self.root / f"budget_{budget}.csv"
            result = score_policy(
                self.run, self.source, path, budget=budget, chunk_size=7
            )
            self.assertEqual(result["selected_n"], expected)
            self.assertEqual(pd.read_csv(path).selected.sum(), expected)
        for budget in (-0.1, 1.1, np.nan, np.inf, True):
            with self.assertRaises(ValueError):
                self._score(budget=budget)

    def test_tied_score_selection_matches_evaluation_for_same_row_order(self):
        #-------------
        # 函数名：ScoringTests.test_tied_score_selection_matches_evaluation_for_same_row_order
        # 功能：验证相同输入行顺序和 seed 下，Score Policy 的同分选择与 Evaluation 排序规则一致。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        self._save_run("t_logistic", _FeatureOnlyModel(constant=True))
        result = self._score(budget=0.2)
        output = pd.read_csv(self.output)
        expected = np.zeros(len(output), dtype=np.int8)
        expected[_rank(output.score.to_numpy(), seed=42)[: result["selected_n"]]] = 1
        np.testing.assert_array_equal(output.selected, expected)

    def test_readonly_input_data_symlink_and_existing_output_guards(self):
        #-------------
        # 函数名：ScoringTests.test_readonly_input_data_symlink_and_existing_output_guards
        # 功能：验证新人群输入只读、符号链接路径保护和拒绝覆盖已有输出文件。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        before = fingerprint(self.source)
        with self.assertRaises(ValueError):
            score_policy(self.run, self.source, self.source)
        protected = self.root / "protected_data"
        protected.mkdir()
        link = self.root / "alias"
        link.symlink_to(protected, target_is_directory=True)
        with patch.object(pipeline_io, "DATA_ROOT", protected):
            for path in (protected / "out.csv", link / "out.csv"):
                with self.assertRaises(ValueError):
                    score_policy(self.run, self.source, path)
        self.output.parent.mkdir()
        self.output.write_text("existing output")
        with self.assertRaises(FileExistsError):
            self._score()
        self.assertEqual(self.output.read_text(), "existing output")
        self.assertEqual(fingerprint(self.source), before)
        self.assertFalse((protected / "out.csv").exists())

    def test_missing_run_model_features_and_unfrozen_selection_have_clear_errors(self):
        #-------------
        # 函数名：ScoringTests.test_missing_run_model_features_and_unfrozen_selection_have_clear_errors
        # 功能：验证缺失 run/模型/特征或未冻结策略时，打分入口返回明确错误。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with self.assertRaisesRegex(FileNotFoundError, "run"):
            score_policy(self.root / "missing", self.source, self.output)
        (self.run / "models" / "t_logistic.joblib").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "模型"):
            self._score()
        self._save_run("random")
        frame = pd.read_csv(self.source).drop(columns="f11")
        frame.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "f11"):
            self._score()
        frame["f11"] = 0
        frame.to_csv(self.source, index=False)
        (self.run / "models" / "selection.json").write_text(
            json.dumps({"selected_policy": "random"})
        )
        with self.assertRaisesRegex(ValueError, "frozen_before_test"):
            self._score()
        (self.run / "models" / "selection.json").write_text(
            json.dumps({"selected_policy": "random", "frozen_before_test": False})
        )
        with self.assertRaisesRegex(ValueError, "frozen_before_test"):
            self._score()
        self.assertFalse(self.output.exists())

    def test_invalid_features_and_excessive_chunk_size_fail_before_output(self):
        #-------------
        # 函数名：ScoringTests.test_invalid_features_and_excessive_chunk_size_fail_before_output
        # 功能：验证非法特征值或超出上限的 chunk_size 会在创建输出文件前失败。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        self._save_run("random")
        frame = pd.read_csv(self.source)
        frame.loc[10, "f0"] = np.nan
        frame.to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "前置特征"):
            self._score()
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            score_policy(self.run, self.source, self.output, chunk_size=100_001)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
