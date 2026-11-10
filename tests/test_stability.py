# 项目测试：验证开发集敏感性分析。

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from criteo_uplift.config import FEATURES
from criteo_uplift.pipeline_io import fingerprint
from criteo_uplift.pipeline_io import save_sample, write_result
from criteo_uplift.models import make_uplift_model
from criteo_uplift.stability import development_splits, feature_groups, load_development_sample, nested_stratified_indices, row_ids_sha256, run_stability


#-----------------------------------------------------------------------------------------------------------------------
# 函数：synthetic_inputs —— 创建敏感性分析测试需要的缓存和来源记录。
# 输入：
#     root：项目根目录，用于定位缓存或结果。
#     n：样本数量或预期数组长度。
# 输出：
#     cache：合成样本缓存路径。
#     base：模拟原实验目录。
# 说明：创建敏感性分析测试需要的缓存和来源记录。
#-----------------------------------------------------------------------------------------------------------------------
def synthetic_inputs(root, n=2000):
    rng = np.random.default_rng(91)
    # 构造重复的特征行，使组间泄漏检查确实能够发现问题。
    x = np.repeat(rng.normal(size=(n // 2, 12)), 2, axis=0).astype(np.float32)
    frame = pd.DataFrame(x, columns=FEATURES)
    frame.insert(0, "row_id", np.arange(n, dtype=np.int64))
    frame["treatment"] = rng.binomial(1, 0.7, n).astype(np.int8)
    frame["conversion"] = rng.binomial(
        1, 0.15 + 0.08 * (x[:, 0] > 0) * frame.treatment, n
    ).astype(np.int8)
    frame["visit"] = np.maximum(frame.conversion, rng.binomial(1, 0.3, n)).astype(
        np.int8
    )
    frame["exposure"] = frame.treatment
    cache = root / "cache/sample.npz"
    base = root / "base"
    base.mkdir()
    (base / "data_preparation").mkdir()
    save_sample(frame, cache)
    write_result(
        {"source": {"sha256": "synthetic_source"}, "sample_size": n, "source_n": n, "sample_n": n},
        cache.with_suffix(".json"),
    )
    write_result(
        {
            "status": "complete",
            "source": {"sha256": "synthetic_source"},
            "sample_n": n,
            "primary_outcome": "conversion",
        },
        base / "manifest.json",
    )
    frame.iloc[::5][["row_id"]].to_csv(base / "data_preparation" / "test_row_ids.csv", index=False)
    return cache, base


#-----------------------------------------------------------------------------------------------------------------------
# 类：StabilityTests —— 验证开发集敏感性分析。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证开发集敏感性分析。
#-----------------------------------------------------------------------------------------------------------------------
class StabilityTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_nested_proportional_sampling —— 核对按比例分层的嵌套训练抽样。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对按比例分层的嵌套训练抽样。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_nested_proportional_sampling(self):
        t = np.repeat([0, 0, 1, 1], [19, 3, 31, 7])
        y = np.repeat([0, 1, 0, 1], [19, 3, 31, 7])
        subsets = nested_stratified_indices(t, y, [0.25, 0.5, 1.0], 17)
        for lower, upper in [(0.25, 0.5), (0.5, 1.0)]:
            self.assertTrue(set(subsets[lower]).issubset(subsets[upper]))
        for fraction, selected in subsets.items():
            self.assertEqual(len(selected), len(np.unique(selected)))
            for stratum in range(4):
                self.assertEqual(
                    int(((2 * t + y)[selected] == stratum).sum()),
                    int(np.floor(fraction * ((2 * t + y) == stratum).sum())),
                )
        np.testing.assert_array_equal(
            subsets[0.5], nested_stratified_indices(t, y, [0.5], 17)[0.5]
        )
        self.assertEqual(len(subsets[1.0]), len(t))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_sampling_inputs —— 确认嵌套抽样拒绝非法输入。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认嵌套抽样拒绝非法输入。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_sampling_inputs(self):
        for fractions in ([], [0], [1.1], [0.5, 0.5], [1, 0.5], [np.nan]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                nested_stratified_indices([0, 1], [0, 1], fractions)
        for t, y in [([1, 1], [0, 1]), ([0, 1], [0, 2]), ([0, 1], [1]), ([], [])]:
            with self.assertRaises(ValueError):
                nested_stratified_indices(t, y)
        with self.assertRaises(ValueError):
            nested_stratified_indices([0, 1], [0, 1], [0.1])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_old_test_excluded_and_group_split_has_zero_overlap —— 确认排除原测试集且特征分组划分无组间重叠。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认排除原测试集且特征分组划分无组间重叠。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_old_test_excluded_and_group_split_has_zero_overlap(self):
        with TemporaryDirectory() as folder:
            cache, base = synthetic_inputs(Path(folder))
            development, old_test, provenance = load_development_sample(cache, base)
            self.assertEqual(len(development), 1600)
            self.assertEqual(provenance["development_old_test_overlap"], 0)
            self.assertFalse(np.isin(development.row_id, old_test).any())
            groups = feature_groups(development)
            splits = development_splits(development, seed=17)
            for method, (train, val) in splits.items():
                self.assertEqual(len(set(train) & set(val)), 0)
                self.assertEqual(len(train) + len(val), len(development))
                if method == "feature_group":
                    self.assertEqual(len(set(groups[train]) & set(groups[val])), 0)
                else:
                    self.assertGreater(len(set(groups[train]) & set(groups[val])), 0)
            self.assertEqual(row_ids_sha256([9, 2, 1]), row_ids_sha256([1, 9, 2]))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_requested_sample_larger_than_source_uses_audited_row_count —— 核验目标样本量超过来源总量时的缓存。
    # 输入：
    #     self：当前测试对象。
    # 输出：
    #     返回值：None；通过断言验证合法全量缓存和不一致缓存的行为。
    # 说明：完整保留来源记录的缓存可以使用，截断或缺少审计依据的缓存仍应报错。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_requested_sample_larger_than_source_uses_source_row_count(self):
        with TemporaryDirectory() as folder:
            cache, base = synthetic_inputs(Path(folder))
            metadata_path = cache.with_suffix('.json')
            metadata = json.loads(metadata_path.read_text())
            write_result({**metadata, 'sample_size': 5000}, metadata_path)
            development, old_test, _ = load_development_sample(cache, base)
            self.assertEqual(len(development) + len(old_test), 2000)
            for count_fields in ({'source_n': 3000}, {'source_n': None}, {'sample_n': 1999}):
                write_result({**metadata, 'sample_size': 5000, **count_fields}, metadata_path)
                with self.assertRaisesRegex(ValueError, 'cache metadata'):
                    load_development_sample(cache, base)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_bad_provenance_and_unknown_test_id_rejected —— 确认敏感性分析拒绝不一致的来源和未知测试行号。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认敏感性分析拒绝不一致的来源和未知测试行号。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_bad_provenance_and_unknown_test_id_rejected(self):
        with TemporaryDirectory() as folder:
            cache, base = synthetic_inputs(Path(folder))
            metadata = json.loads(cache.with_suffix(".json").read_text())
            metadata["source"]["sha256"] = "other"
            write_result(metadata, cache.with_suffix(".json"))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_development_sample(cache, base)
            metadata["source"]["sha256"] = "synthetic_source"
            write_result(metadata, cache.with_suffix(".json"))
            pd.DataFrame({"row_id": [999999]}).to_csv(
                base / "data_preparation" / "test_row_ids.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "not all present"):
                load_development_sample(cache, base)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_small_integration_freezes_protocol_and_preserves_inputs —— 确认小规模敏感性流程冻结配置并保留输入。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认小规模敏感性流程冻结配置并保留输入。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_small_integration_freezes_protocol_and_preserves_inputs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache, base = synthetic_inputs(root)
            output = root / "study"
            paths = [
                cache,
                cache.with_suffix(".json"),
                base / "manifest.json",
                base / "data_preparation" / "test_row_ids.csv",
            ]
            before = {str(p): fingerprint(p) for p in paths}
            original_fit_calls = []

            #-----------------------------------------------------------------------------------------------------------------------
            # 函数：checked_make_model —— 创建并包装模型以记录拟合数据。
            # 输入：
            #     name：需要构造、读取或保存的对象名称。
            #     **kwargs：传递给被调用函数的额外命名参数。
            # 输出：
            #     model：带拟合数据记录功能的真实模型。
            # 说明：创建并包装模型以记录拟合数据。
            #-----------------------------------------------------------------------------------------------------------------------
            def checked_make_model(**kwargs):
                name = kwargs["learner"] + "_" + kwargs["outcome_model"]
                model = make_uplift_model(**kwargs)
                original_fit = model.fit

                #-----------------------------------------------------------------------------------------------------------------------
                # 函数：fit —— 使用输入数据拟合当前模型。
                # 输入：
                #     X：处理前特征矩阵，行对应样本，列对应特征。
                #     t：每行的处理组标签，0 为 Control，1 为 Treatment。
                #     y：每行的二分类结果标签。
                # 输出：
                #     self：完成拟合的当前模型对象。
                # 说明：使用输入数据拟合当前模型。
                #-----------------------------------------------------------------------------------------------------------------------
                def fit(X, t, y):
                    self.assertTrue((output / "protocol.json").exists())
                    protocol = json.loads((output / "protocol.json").read_text())
                    self.assertEqual(protocol["status"], "frozen_before_model_fits")
                    original_fit_calls.append(name)
                    return original_fit(X, t, y)

                model.fit = fit
                return model

            with patch(
                "criteo_uplift.stability.make_uplift_model", side_effect=checked_make_model
            ):
                result = run_stability(
                    cache, base, output, seeds=[17], fractions=[0.5, 1.0], trees=3
                )
            self.assertEqual(result, output.resolve())
            self.assertEqual(len(original_fit_calls), 12)
            metrics = pd.read_csv(output / "longmetrics.csv")
            self.assertEqual(len(metrics), 18)
            self.assertTrue(
                {"qini_ht", "qini_score_traditional", "auuc_ht"}.issubset(
                    metrics.columns
                )
            )
            self.assertEqual(metrics.selected_on_validation.sum(), 3)
            self.assertTrue(metrics[metrics.policy == "random"].qini_ht.eq(0).all())
            audit = pd.read_csv(output / "split_audit.csv")
            self.assertTrue(
                audit[
                    [
                        "train_old_test_overlap",
                        "validation_old_test_overlap",
                        "row_overlap",
                    ]
                ]
                .eq(0)
                .all()
                .all()
            )
            self.assertTrue(
                audit[audit.split_method == "feature_group"]
                .feature_group_overlap.eq(0)
                .all()
            )
            for method, subset in audit.groupby("split_method"):
                self.assertEqual(subset.assignment_probability.nunique(), 1)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["old_test_n_excluded"], 400)
            self.assertFalse(report["original_test_re_evaluated"])
            self.assertFalse(list(output.glob("*.joblib")))
            for path in paths:
                self.assertEqual(fingerprint(path), before[str(path)])
            with self.assertRaises(FileExistsError):
                run_stability(cache, base, output, seeds=[17], fractions=[1.0], trees=3)
            with self.assertRaises(ValueError):
                run_stability(
                    cache, base, base / "nested", seeds=[17], fractions=[1.0], trees=3
                )
            for kwargs in (
                {"seeds": [17, 17]},
                {"seeds": [-1]},
                {"trees": 0},
                {"primary_budget": 0},
            ):
                with self.assertRaises(ValueError):
                    run_stability(cache, base, root / "unused", **kwargs)


if __name__ == "__main__":
    unittest.main()
