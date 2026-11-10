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
from criteo_uplift.preparation import _scan_raw_data


#-----------------------------------------------------------------------------------------------------------------------
# 类：DiagnosticsTests —— 验证全量诊断与 C2ST。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证全量诊断与 C2ST。
#-----------------------------------------------------------------------------------------------------------------------
class DiagnosticsTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_smd_matches_pooled_standard_deviation —— 核对 SMD 使用的合并标准差。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对 SMD 使用的合并标准差。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_smd_matches_pooled_standard_deviation(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_post_treatment_features_are_rejected —— 确认诊断拒绝处理后的特征。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认诊断拒绝处理后的特征。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_post_treatment_features_are_rejected(self):
        frame = pd.DataFrame({"exposure": [0, 0, 1, 1], "treatment": [0, 0, 1, 1]})
        with self.assertRaisesRegex(ValueError, "pre-treatment"):
            covariate_balance(frame, ["exposure"])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_srm_defaults_to_published_reference_and_accepts_other_ratios —— 核对 SRM 默认分配比例和自定义比例。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对 SRM 默认分配比例和自定义比例。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_srm_defaults_to_published_reference_and_accepts_other_ratios(self):
        reference = srm_summary(850, 150)
        self.assertEqual(reference["status"], "computed")
        self.assertEqual(reference["expected_ratio"], 0.85)
        self.assertAlmostEqual(reference["chi2"], 0.0)
        self.assertAlmostEqual(reference["pvalue"], 1.0)
        balanced = srm_summary(50, 50, expected_ratio=0.5)
        self.assertEqual(balanced["chi2"], 0.0)
        self.assertEqual(balanced["pvalue"], 1.0)
        self.assertLess(srm_summary(850, 150, expected_ratio=0.5)["pvalue"], 1e-50)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_srm_rejects_missing_or_invalid_reference_ratios —— 确认 SRM 拒绝缺失或非法参考比例。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认 SRM 拒绝缺失或非法参考比例。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_srm_rejects_missing_or_invalid_reference_ratios(self):
        for ratio in (None, 0.0, 1.0, np.nan, np.inf):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    srm_summary(850, 150, expected_ratio=ratio)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_c2st_complete_refits_and_permutation_pvalues —— 核对 C2ST 完整重拟合次数和置换 p 值。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对 C2ST 完整重拟合次数和置换 p 值。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_c2st_complete_refits_and_permutation_pvalues(self):
        rng = np.random.default_rng(2026)
        t = np.tile([0, 1], 150)
        frame = pd.DataFrame({"treatment": t, "f0": 4 * t + rng.normal(size=len(t))})
        original = frame.copy(deep=True)
        from criteo_uplift import diagnostics

        factory = diagnostics._make_c2st_model
        signatures = []
        def record_model(name, seed):
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
        with self.assertRaisesRegex(ValueError, "Unknown model"):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_learnability_dummy_uses_training_rate —— 确认可学习性检查的常数基线仅使用训练集正例率。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认可学习性检查的常数基线仅使用训练集正例率。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_learnability_dummy_uses_training_rate(self):
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_single_class_outcomes_are_explicit —— 确认单类别结果的指标限制被明确记录。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认单类别结果的指标限制被明确记录。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_single_class_outcomes_are_explicit(self):
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
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：frame —— 创建用于诊断验证的合成数据。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     frame：包含特征和实验标签的合成数据表。
    # 说明：创建用于诊断验证的合成数据。
    #-----------------------------------------------------------------------------------------------------------------------
    def frame(self):
        rng = np.random.default_rng(6)
        frame = pd.DataFrame(rng.normal(size=(80, 12)), columns=FEATURES)
        # 构造不等大的两组，并包含常量及 float32 下难以区分的近常量特征。
        frame["f0"] = 7.0
        frame["f1"] = 1 + np.arange(80) * 1e-10
        for name in LABELS:
            frame[name] = rng.integers(0, 2, 80)
        frame["treatment"] = np.r_[np.zeros(20), np.ones(60)]
        return frame[COLUMNS]

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_full_stats_balanced_sampling_precision_and_chunk_reproducibility —— 核对全量统计、平衡抽样、数值精度和分块可复现性。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对全量统计、平衡抽样、数值精度和分块可复现性。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_full_stats_balanced_sampling_precision_and_chunk_reproducibility(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            self.frame().to_csv(path, index=False)
            original = path.read_bytes()
            raw = pd.read_csv(path, dtype=np.float64)
            expected = covariate_balance(raw, FEATURES)
            scanned = _scan_raw_data(path, sample_size=None, per_arm=10, chunk_size=13, progress=lambda _: None)
            first, sample, meta = scanned["balance"], scanned["c2st_sample"], scanned["metadata"]
            scanned = _scan_raw_data(path, sample_size=None, per_arm=10, chunk_size=31, progress=lambda _: None)
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_data_fails_without_silent_row_deletion —— 确认非法数据报错而非静默删行。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认非法数据报错而非静默删行。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_data_fails_without_silent_row_deletion(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            frame = self.frame()
            frame.loc[0, "f0"] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "[Nn]onfinite"):
                _scan_raw_data(path, sample_size=None, per_arm=10, progress=lambda _: None)
            frame = self.frame()
            frame.loc[0, "treatment"] = 2
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "binary"):
                _scan_raw_data(path, sample_size=None, per_arm=10, progress=lambda _: None)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_update_replaces_only_existing_diagnostic_files —— 确认诊断更新只替换对应的诊断产物。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认诊断更新只替换对应的诊断产物。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_update_replaces_only_existing_diagnostic_files(self):
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
                scanned=_scan_raw_data(raw, sample_size=None, per_arm=10, progress=lambda _: None), audit={'n_treatment': 60, 'n_control': 20},
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
