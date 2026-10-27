# 项目测试：验证冻结策略离线打分。

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from criteo_uplift import data, pipeline_io
from criteo_uplift.data import FEATURES, fingerprint
from criteo_uplift.evaluation import _rank
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：__init__ —— 保存模型或测试对象的初始化配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     constant：是否让测试模型输出相同分数。
    #     max_rows：测试模型允许的每批最大行数。
    # 输出：
    #     返回值：None；参数保存在当前对象中。
    # 说明：保存模型或测试对象的初始化配置。
    #-----------------------------------------------------------------------------------------------------------------------
    def __init__(self, constant=False, max_rows=7):
        self.constant = constant
        self.max_rows = max_rows

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_check_features —— 检查测试模型收到的特征结构和分块规模。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     返回值：None；不符合约定时触发断言。
    # 说明：检查测试模型收到的特征结构和分块规模。
    #-----------------------------------------------------------------------------------------------------------------------
    def _check_features(self, X):
        if list(X.columns) != FEATURES or len(X) > self.max_rows:
            raise AssertionError(
                "Only ordered pre-treatment features in bounded chunks are allowed."
            )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict —— 按测试配置返回常数或第一列特征。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     分数数组：与输入行顺序一致的测试分数。
    # 说明：按测试配置返回常数或第一列特征。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict(self, X):
        self._check_features(X)
        return np.ones(len(X)) if self.constant else X.f0.to_numpy()

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：predict_outcomes —— 生成用于区分打分路径的测试输出。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     X：处理前特征矩阵，行对应样本，列对应特征。
    # 输出：
    #     mu0：第一列测试分数的两倍。
    #     mu1：第一列测试分数。
    # 说明：生成用于区分打分路径的测试输出。
    #-----------------------------------------------------------------------------------------------------------------------
    def predict_outcomes(self, X):
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
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：tearDown —— 清理当前测试的临时资源。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；删除测试创建的临时目录。
    # 说明：清理当前测试的临时资源。
    #-----------------------------------------------------------------------------------------------------------------------
    def tearDown(self):
        self.directory.cleanup()

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_save_run —— 保存供打分测试读取的模拟冻结实验。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     policy：需要使用的策略名称。
    #     model：已经构造或拟合的模型对象。
    # 输出：
    #     返回值：None；写入选择记录、来源记录和可选模型。
    # 说明：保存供打分测试读取的模拟冻结实验。
    #-----------------------------------------------------------------------------------------------------------------------
    def _save_run(self, policy, model=None):
        selection = {"selected_policy": policy, "frozen_before_test": True}
        manifest = {"features": FEATURES, "seed": 42, "selection": selection}
        (self.run / "models").mkdir(exist_ok=True)
        (self.run / "models" / "selection.json").write_text(json.dumps(selection))
        (self.run / "manifest.json").write_text(json.dumps(manifest))
        if policy != "random":
            name = "t_xgboost" if policy == "propensity" else policy
            joblib.dump(model or _FeatureOnlyModel(), self.run / "models" / f"{name}.joblib")

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：_score —— 使用当前临时目录执行打分测试。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     **kwargs：传递给被调用函数的额外命名参数。
    # 输出：
    #     result：离线打分返回的概要字典。
    # 说明：使用当前临时目录执行打分测试。
    #-----------------------------------------------------------------------------------------------------------------------
    def _score(self, **kwargs):
        return score_policy(self.run, self.source, self.output, chunk_size=7, **kwargs)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_feature_only_chunks_exact_budget_and_input_unchanged —— 确认分块打分只读取特征、准确执行预算且保留输入。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认分块打分只读取特征、准确执行预算且保留输入。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_feature_only_chunks_exact_budget_and_input_unchanged(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_propensity_uses_treated_response_probability —— 确认响应概率基线使用处理状态下的结果概率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认响应概率基线使用处理状态下的结果概率。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_propensity_uses_treated_response_probability(self):
        self._save_run("propensity")
        self._score(budget=0.2)
        output = pd.read_csv(self.output)
        np.testing.assert_allclose(output.score, np.arange(23) / 100.0)
        np.testing.assert_array_equal(
            output.loc[output.selected == 1, "source_row_id"], [19, 20, 21, 22]
        )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_random_scores_and_tie_breaking_are_chunk_independent —— 确认随机策略和同分排序不受分块大小影响。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认随机策略和同分排序不受分块大小影响。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_random_scores_and_tie_breaking_are_chunk_independent(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_budget_boundaries_and_decimal_floor —— 核对预算边界及十进制预算取整。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对预算边界及十进制预算取整。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_budget_boundaries_and_decimal_floor(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_tied_score_selection_matches_evaluation_for_same_row_order —— 确认离线打分和评价对相同输入的同分选择一致。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认离线打分和评价对相同输入的同分选择一致。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_tied_score_selection_matches_evaluation_for_same_row_order(self):
        self._save_run("t_logistic", _FeatureOnlyModel(constant=True))
        result = self._score(budget=0.2)
        output = pd.read_csv(self.output)
        expected = np.zeros(len(output), dtype=np.int8)
        expected[_rank(output.score.to_numpy(), seed=42)[: result["selected_n"]]] = 1
        np.testing.assert_array_equal(output.selected, expected)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_readonly_input_data_symlink_and_existing_output_guards —— 核对只读输入、数据目录链接及已有输出的保护。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对只读输入、数据目录链接及已有输出的保护。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_readonly_input_data_symlink_and_existing_output_guards(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_missing_run_model_features_and_unfrozen_selection_have_clear_errors —— 确认缺少实验、模型、特征或冻结记录时明确报错。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认缺少实验、模型、特征或冻结记录时明确报错。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_missing_run_model_features_and_unfrozen_selection_have_clear_errors(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_features_and_excessive_chunk_size_fail_before_output —— 确认非法特征及过大分块在写出结果前被拒绝。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认非法特征及过大分块在写出结果前被拒绝。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_features_and_excessive_chunk_size_fail_before_output(self):
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
