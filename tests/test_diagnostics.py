# 项目测试：验证全量诊断与 C2ST。

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from criteo_uplift.config import COLUMNS, FEATURES, LABELS
from criteo_uplift.pipeline_io import fingerprint

from criteo_uplift.diagnostics import c2st, covariate_balance, learnability, srm_summary, run_diagnostics
from criteo_uplift.raw_scan import scan_raw_data


#-----------------------------------------------------------------------------------------------------------------------
# 类：DiagnosticsTests —— 验证全量诊断与 C2ST。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证全量诊断与 C2ST。
#-----------------------------------------------------------------------------------------------------------------------
class DiagnosticsTests(unittest.TestCase):
    def test_smd_matches_pooled_standard_deviation(self):
        #-------------
        # 函数名：DiagnosticsTests.test_smd_matches_pooled_standard_deviation
        # 功能：验证协变量平衡函数的 SMD 与 pooled 样本标准差手工计算结果一致。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = pd.DataFrame(
            {
                "f0": [1.0, 3.0, 2.0, 4.0],
                "f1": [7.0, 7.0, 7.0, 7.0],
                "treatment": [0, 0, 1, 1],
            }
        )
        original = frame.copy(deep=True)
        result = covariate_balance(frame, ["f0", "f1"]).set_index("feature")
        self.assertAlmostEqual(result.loc["f0", "smd"], 1 / np.sqrt(2))
        self.assertEqual(result.loc["f1", "smd"], 0.0)
        self.assertEqual(result.loc["f1", "ks_statistic"], 0.0)
        assert_frame_equal(frame, original)

    def test_post_treatment_features_are_rejected(self):
        #-------------
        # 函数名：DiagnosticsTests.test_post_treatment_features_are_rejected
        # 功能：验证 C2ST/平衡诊断拒绝 exposure 等处理后变量作为处理前特征。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = pd.DataFrame({"exposure": [0, 0, 1, 1], "treatment": [0, 0, 1, 1]})
        with self.assertRaisesRegex(ValueError, "前置特征"):
            covariate_balance(frame, ["exposure"])

    def test_srm_defaults_to_published_reference_and_accepts_other_ratios(self):
        #-------------
        # 函数名：DiagnosticsTests.test_srm_defaults_to_published_reference_and_accepts_other_ratios
        # 功能：验证 SRM 默认采用 0.85 公开参考比例，并可显式使用其他参考比例。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        reference = srm_summary(850, 150)
        self.assertEqual(reference["status"], "computed")
        self.assertEqual(reference["expected_ratio"], 0.85)
        self.assertAlmostEqual(reference["chi2"], 0.0)
        self.assertAlmostEqual(reference["pvalue"], 1.0)
        balanced = srm_summary(50, 50, expected_ratio=0.5)
        self.assertEqual(balanced["chi2"], 0.0)
        self.assertEqual(balanced["pvalue"], 1.0)
        self.assertLess(srm_summary(850, 150, expected_ratio=0.5)["pvalue"], 1e-50)

    def test_srm_rejects_missing_or_invalid_reference_ratios(self):
        #-------------
        # 函数名：DiagnosticsTests.test_srm_rejects_missing_or_invalid_reference_ratios
        # 功能：验证 SRM 对缺失、非有限或不在 (0,1) 内的参考比例明确报错。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for ratio in (None, 0.0, 1.0, np.nan, np.inf):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    srm_summary(850, 150, expected_ratio=ratio)

    def test_c2st_complete_refits_and_permutation_pvalues(self):
        #-------------
        # 函数名：DiagnosticsTests.test_c2st_complete_refits_and_permutation_pvalues
        # 功能：验证 C2ST 每轮置换均重新拟合分类器，且置换 p 值按有限次校正公式计算。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        rng = np.random.default_rng(2026)
        t = np.tile([0, 1], 150)
        frame = pd.DataFrame({"treatment": t, "f0": 4 * t + rng.normal(size=len(t))})
        original = frame.copy(deep=True)
        from criteo_uplift import diagnostics

        factory = diagnostics._make_c2st_model
        signatures = []
        def record_model(name, seed):
            #-------------
            # 函数名：DiagnosticsTests.test_c2st_complete_refits_and_permutation_pvalues.record_model
            # 功能：记录 C2ST 测试中每次模型工厂调用的模型名称和随机种子，并返回可记录拟合的测试分类器。
            # 输入参数：
            #     - name：本次请求创建的分类器名称。
            #     - seed：传给分类器的随机种子。
            # 输出参数：
            #     - 测试用模型实例。
            #-------------
            model = factory(name, seed)
            signatures.append((name, {key: repr(value) for key, value in model.get_params().items()}))
            return model
        with patch.object(diagnostics, "_make_c2st_model", side_effect=record_model) as wrapped:
            result = c2st(frame, ["f0"], sample_size=300, permutations=3, seed=7)
        for name in ("logistic", "xgboost"):
            configurations = [params for model_name, params in signatures if model_name == name]
            self.assertEqual(len(configurations), 4)
            self.assertTrue(all(params == configurations[0] for params in configurations))
        repeat = c2st(frame, ["f0"], sample_size=300, permutations=3, seed=7)
        assert_frame_equal(result["summary"], repeat["summary"])
        assert_frame_equal(result["null_auc"], repeat["null_auc"])
        self.assertTrue({"log_loss", "brier", "average_precision"}.issubset(result["summary"].columns))
        with self.assertRaisesRegex(ValueError, "不支持的模型"):
            factory("hgb", 7)
        self.assertEqual(wrapped.call_count, 2 * (1 + 3))
        self.assertEqual(set(result["summary"].model), {"logistic", "xgboost"})
        self.assertEqual(factory("xgboost", 7).get_params()["n_estimators"], 30)
        self.assertEqual(factory("xgboost", 7).get_params()["tree_method"], "hist")
        self.assertEqual(result["train_n"] + result["test_n"], 300)
        self.assertEqual(len(result["null_auc"]), 6)
        for row in result["summary"].itertuples():
            null = result["null_auc"].query("model == @row.model")["roc_auc"]
            expected = (1 + (null >= row.roc_auc).sum()) / 4
            self.assertEqual(row.pvalue, expected)
            self.assertGreaterEqual(row.pvalue_holm, row.pvalue)
            self.assertGreater(row.roc_auc, 0.90)
        assert_frame_equal(frame, original)

    def test_learnability_dummy_uses_training_rate(self):
        #-------------
        # 函数名：DiagnosticsTests.test_learnability_dummy_uses_training_rate
        # 功能：验证 Learnability 的 Dummy 概率基线使用训练集正例率，而不是验证集正例率。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        train = pd.DataFrame(
            {
                "f0": np.arange(100, dtype=float),
                "treatment": np.tile([0, 1], 50),
                "conversion": np.r_[np.ones(20), np.zeros(80)],
            }
        )
        val = pd.DataFrame(
            {
                "f0": np.arange(20, dtype=float),
                "treatment": np.tile([0, 1], 10),
                "conversion": np.tile([0, 1], 10),
            }
        )
        result = learnability(train, val, ["f0"], "conversion", by_arm=False)
        dummy = result.set_index("model").loc["dummy"]
        self.assertAlmostEqual(dummy["log_loss"], -(np.log(0.2) + np.log(0.8)) / 2)
        self.assertAlmostEqual(dummy["brier"], (0.2**2 + 0.8**2) / 2)
        self.assertEqual(dummy["average_precision"], 0.5)

    def test_single_class_outcomes_are_explicit(self):
        #-------------
        # 函数名：DiagnosticsTests.test_single_class_outcomes_are_explicit
        # 功能：验证训练集单类别结果变量会被显式标记或跳过，不会产生伪造的分类指标。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = pd.DataFrame(
            {
                "f0": np.arange(20, dtype=float),
                "treatment": np.tile([0, 1], 10),
                "conversion": np.zeros(20, dtype=int),
            }
        )
        result = learnability(frame.iloc[:10], frame.iloc[10:], ["f0"], "conversion")
        dummy = result[result["model"] == "dummy"]
        fitted = result[result["model"] != "dummy"]
        self.assertTrue(dummy["roc_auc"].isna().all())
        self.assertTrue(dummy["average_precision"].isna().all())
        self.assertTrue((fitted["status"] == "skipped_single_class_train").all())


#-----------------------------------------------------------------------------------------------------------------------
# 类：PublicDiagnosticsTests —— 验证全量诊断与 C2ST。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证全量诊断与 C2ST。
#-----------------------------------------------------------------------------------------------------------------------
class PublicDiagnosticsTests(unittest.TestCase):
    def frame(self):
        #-------------
        # 函数名：PublicDiagnosticsTests.frame
        # 功能：构造包含处理前特征、处理分配和结果标签的公共诊断合成数据。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 诊断测试使用的 pandas DataFrame。
        #-------------
        rng = np.random.default_rng(6)
        frame = pd.DataFrame(rng.normal(size=(80, 12)), columns=FEATURES)
        # 构造不等大的两组，并包含常量及 float32 下难以区分的近常量特征。
        frame["f0"] = 7.0
        frame["f1"] = 1 + np.arange(80) * 1e-10
        for name in LABELS:
            frame[name] = rng.integers(0, 2, 80)
        frame["treatment"] = np.r_[np.zeros(20), np.ones(60)]
        return frame[COLUMNS]

    def test_full_stats_balanced_sampling_precision_and_chunk_reproducibility(self):
        #-------------
        # 函数名：PublicDiagnosticsTests.test_full_stats_balanced_sampling_precision_and_chunk_reproducibility
        # 功能：验证全量统计、处理臂等量抽样、数值精度及不同分块大小下的可复现性。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            self.frame().to_csv(path, index=False)
            original = path.read_bytes()
            raw = pd.read_csv(path, dtype=np.float64)
            expected = covariate_balance(raw, FEATURES)
            scanned = scan_raw_data(path, sample_size=None, per_arm=10, chunk_size=13, progress=lambda _: None)
            first, sample, meta = scanned["balance"], scanned["c2st_sample"], scanned["metadata"]
            scanned = scan_raw_data(path, sample_size=None, per_arm=10, chunk_size=31, progress=lambda _: None)
            second, repeated = scanned["balance"], scanned["c2st_sample"]
            assert_frame_equal(first[expected.columns], expected)
            assert_frame_equal(first, second)
            assert_frame_equal(sample, repeated)
            self.assertEqual(sample.treatment.value_counts().to_dict(), {0: 10, 1: 10})
            self.assertEqual(meta["n"], 80)
            self.assertEqual(meta["n_control"], 20)
            self.assertEqual(meta["n_treatment"], 60)
            self.assertEqual(meta["rows_dropped"], 0)
            np.testing.assert_array_equal(sample[FEATURES], raw.iloc[sample.row_id][FEATURES])
            self.assertEqual(sample.f1.dtype, np.dtype("float64"))
            self.assertGreater(sample.f1.nunique(), sample.f1.astype("float32").nunique())
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_data_fails_without_silent_row_deletion(self):
        #-------------
        # 函数名：PublicDiagnosticsTests.test_invalid_data_fails_without_silent_row_deletion
        # 功能：验证原始数据包含非法值时扫描会明确失败，且不会静默删行。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            frame = self.frame()
            frame.loc[0, "f0"] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "非有限值"):
                scan_raw_data(path, sample_size=None, per_arm=10, progress=lambda _: None)
            frame = self.frame()
            frame.loc[0, "treatment"] = 2
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "二元值"):
                scan_raw_data(path, sample_size=None, per_arm=10, progress=lambda _: None)

    def test_update_replaces_only_existing_diagnostic_files(self):
        #-------------
        # 函数名：PublicDiagnosticsTests.test_update_replaces_only_existing_diagnostic_files
        # 功能：验证诊断输出更新只替换已有诊断文件，不误删其他实验产物。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "diagnostics").mkdir()
            (directory / "evaluation").mkdir()
            raw = directory / 'raw.csv'
            frame = self.frame()
            frame.to_csv(raw, index=False)
            diagnostic_names = ['srm.json', 'balance.csv', 'c2st.csv', 'c2st_null.csv', 'c2st_details.json', 'learnability.csv']
            for name in diagnostic_names:
                (directory / 'diagnostics' / name).write_text('old diagnostic')
            protected = (directory / 'evaluation' / 'predictions.npz')
            protected.write_bytes(b'fixed-model-predictions')
            before_names = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
            details = run_diagnostics(
                directory / "diagnostics", train=frame, validation=frame,
                scanned=scan_raw_data(raw, sample_size=None, per_arm=10, progress=lambda _: None), audit={'n_treatment': 60, 'n_control': 20},
                source=fingerprint(raw), sample_size=20, permutations=3,
                progress=lambda _: None,
            )
            self.assertEqual({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}, before_names)
            self.assertEqual(protected.read_bytes(), b'fixed-model-predictions')
            self.assertEqual(details['full_n'], 80)
            self.assertEqual(details['n_control'], 10)
            self.assertEqual(details['n_treatment'], 10)
            self.assertTrue(pd.read_csv((directory / 'diagnostics' / 'balance.csv'))['scope'].eq('full_published_dataset').all())
            self.assertEqual(len(pd.read_csv((directory / 'diagnostics' / 'c2st_null.csv'))), 6)
            self.assertEqual(json.loads((directory / 'diagnostics' / 'srm.json').read_text())['expected_ratio'], .85)


if __name__ == "__main__":
    unittest.main()
