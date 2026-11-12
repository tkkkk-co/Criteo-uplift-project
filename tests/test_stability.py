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
from functools import partial
from criteo_uplift.models import LEARNERS
from criteo_uplift.diagnostics import diagnose_overlap
from criteo_uplift.stability import development_splits, feature_groups, load_development_sample, nested_stratified_indices, row_ids_sha256, run_stability


def synthetic_inputs(root, n=2000):
    #-------------
    # 函数名：synthetic_inputs
    # 功能：生成 Stability 测试用合成样本、NPZ 缓存、metadata、基准 manifest 和旧 Test row_id 文件。
    # 输入参数：
    #     - root：测试使用的临时项目根目录。
    #     - n：合成样本行数。
    # 输出参数：
    #     - 二元组：(合成样本缓存路径、模拟基准实验目录路径)。
    #-------------
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
    def test_nested_proportional_sampling(self):
        #-------------
        # 函数名：StabilityTests.test_nested_proportional_sampling
        # 功能：验证嵌套分层抽样按各 T×Y 分层比例抽样、无放回且不同训练比例之间保持嵌套。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_invalid_sampling_inputs(self):
        #-------------
        # 函数名：StabilityTests.test_invalid_sampling_inputs
        # 功能：验证嵌套抽样拒绝空比例、越界比例、重复/乱序比例和非法随机种子。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for fractions in ([], [0], [1.1], [0.5, 0.5], [1, 0.5], [np.nan]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                nested_stratified_indices([0, 1], [0, 1], fractions)
        for t, y in [([1, 1], [0, 1]), ([0, 1], [0, 2]), ([0, 1], [1]), ([], [])]:
            with self.assertRaises(ValueError):
                nested_stratified_indices(t, y)
        with self.assertRaises(ValueError):
            nested_stratified_indices([0, 1], [0, 1], [0.1])

    def test_old_test_excluded_and_group_split_has_zero_overlap(self):
        #-------------
        # 函数名：StabilityTests.test_old_test_excluded_and_group_split_has_zero_overlap
        # 功能：验证 Stability 开发样本排除原 Test，并在匿名特征分组划分中保持组间零重叠。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_requested_sample_larger_than_source_uses_source_row_count(self):
        #-------------
        # 函数名：StabilityTests.test_requested_sample_larger_than_source_uses_source_row_count
        # 功能：验证缓存抽样目标大于原始数据量时，按真实来源行数核验样本规模。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with TemporaryDirectory() as folder:
            cache, base = synthetic_inputs(Path(folder))
            metadata_path = cache.with_suffix('.json')
            metadata = json.loads(metadata_path.read_text())
            write_result({**metadata, 'sample_size': 5000}, metadata_path)
            development, old_test, _ = load_development_sample(cache, base)
            self.assertEqual(len(development) + len(old_test), 2000)
            for count_fields in ({'source_n': 3000}, {'source_n': None}, {'sample_n': 1999}):
                write_result({**metadata, 'sample_size': 5000, **count_fields}, metadata_path)
                with self.assertRaisesRegex(ValueError, '缓存说明'):
                    load_development_sample(cache, base)

    def test_bad_provenance_and_unknown_test_id_rejected(self):
        #-------------
        # 函数名：StabilityTests.test_bad_provenance_and_unknown_test_id_rejected
        # 功能：验证来源 SHA 不一致或旧 Test row_id 不存在于缓存时拒绝开展 Stability。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
            with self.assertRaisesRegex(ValueError, "未包含原测试集"):
                load_development_sample(cache, base)

    def test_small_integration_freezes_protocol_and_preserves_inputs(self):
        #-------------
        # 函数名：StabilityTests.test_small_integration_freezes_protocol_and_preserves_inputs
        # 功能：验证小规模 Stability 集成流程先冻结 protocol、排除旧 Test、完成模型比较且不修改只读输入。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
            learners = LEARNERS.copy()

            def checked_learner(learner, X, **kwargs):
                #-------------
                # 确认训练前已冻结协议，并记录每次拟合。
                # 输入：Learner 类型、预测特征 X、训练参数。
                # 输出：拟合和预测结果。
                #-------------
                self.assertTrue((output / 'protocol.json').exists())
                protocol = json.loads((output / 'protocol.json').read_text())
                self.assertEqual(protocol['status'], 'frozen_before_model_fits')
                self.assertIn('X_train', kwargs)
                original_fit_calls.append(learner + '_' + kwargs['outcome_model'])
                return learners[learner](X, **kwargs)

            with patch.dict(
                'criteo_uplift.stability.LEARNERS',
                {name: partial(checked_learner, name) for name in LEARNERS},
            ), patch('criteo_uplift.stability.diagnose_overlap', wraps=diagnose_overlap) as coverage_check:
                result = run_stability(
                    cache, base, output, seeds=[17], fractions=[0.5, 1.0], trees=3
                )
            self.assertEqual(result, output.resolve())
            self.assertEqual(len(original_fit_calls), 12)
            self.assertEqual(coverage_check.call_count, 3)
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
