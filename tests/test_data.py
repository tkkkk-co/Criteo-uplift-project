# 项目测试：验证数据读取、抽样和划分。

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from criteo_uplift import data, pipeline_io
from criteo_uplift.diagnostics import scan_raw_data


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_synthetic_frame —— 创建用于数据读取测试的合成数据。
# 输入：
#     n：样本数量或预期数组长度。
# 输出：
#     frame：具有完整输入列的合成数据表。
# 说明：创建用于数据读取测试的合成数据。
#-----------------------------------------------------------------------------------------------------------------------
def _synthetic_frame(n=400):
    row = np.arange(n)
    frame = pd.DataFrame(
        {feature: row * (j + 1.0) + j / 8.0 for j, feature in enumerate(data.FEATURES)}
    )
    frame["treatment"] = row % 2
    frame["conversion"] = (row // 2) % 2
    frame["visit"] = np.maximum(frame.conversion, row % 5 == 0).astype(int)
    frame["exposure"] = frame.treatment * (row % 3 != 0)
    return frame[data.COLUMNS]


#-----------------------------------------------------------------------------------------------------------------------
# 类：DataTests —— 验证数据读取、抽样和划分。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证数据读取、抽样和划分。
#-----------------------------------------------------------------------------------------------------------------------
class DataTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="criteo_data_test_")
        self.root = Path(self.directory.name)
        self.source = self.root / "synthetic.csv"
        self.frame = _synthetic_frame()
        self.frame.to_csv(self.source, index=False)

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
    # 函数：test_sampling_reads_entire_file_is_reproducible_and_chunk_independent —— 确认抽样读取全量文件且可复现、不受分块影响。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认抽样读取全量文件且可复现、不受分块影响。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_sampling_reads_entire_file_is_reproducible_and_chunk_independent(self):
        before = pipeline_io.fingerprint(self.source)
        scanned = scan_raw_data(
            self.source, sample_size=43, seed=31, chunk_size=57
        )
        first = scanned["sample"]
        audit = scanned["audit"]
        scanned = scan_raw_data(
            self.source, sample_size=43, seed=31, chunk_size=101
        )
        repeated = scanned["sample"]
        scanned = scan_raw_data(
            self.source, sample_size=43, seed=32, chunk_size=57
        )
        different = scanned["sample"]
        pd.testing.assert_frame_equal(first, repeated)
        self.assertFalse(first.row_id.equals(different.row_id))
        self.assertEqual(len(first), 43)
        self.assertEqual(first.row_id.nunique(), 43)
        self.assertTrue(first.row_id.is_monotonic_increasing)
        self.assertGreater(first.row_id.max(), 300)
        np.testing.assert_array_equal(first.f0, first.row_id)
        self.assertEqual(audit["n_rows"], 400)
        self.assertEqual(audit["sample_n"], 43)
        self.assertEqual(audit["n_treatment"], 200)
        self.assertEqual(audit["groups"]["0"]["conversion"]["positive"], 100)
        self.assertEqual(pipeline_io.fingerprint(self.source), before)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_full_data_audit_counts_duplicates_without_removing_rows —— 确认全量审计统计重复行但不删除记录。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认全量审计统计重复行但不删除记录。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_full_data_audit_counts_duplicates_without_removing_rows(self):
        duplicated = pd.concat(
            [self.frame, self.frame.iloc[[7, 7, 210]]], ignore_index=True
        )
        duplicated.to_csv(self.source, index=False)
        scanned = scan_raw_data(self.source, sample_size=0, chunk_size=61)
        sample = scanned["sample"]
        audit = scanned["audit"]
        self.assertEqual(len(sample), len(duplicated))
        self.assertEqual(audit["duplicate_row_fingerprints"], 3)
        self.assertEqual(sample.row_id.nunique(), len(duplicated))
        self.assertEqual(audit["n_treatment"], int(duplicated.treatment.sum()))
        self.assertEqual(audit["control_exposed"], 0)
        self.assertEqual(audit["conversion_without_visit"], 0)
        for feature_stats in audit["feature_stats"]:
            column = feature_stats["feature"]
            for arm in (0, 1):
                values = duplicated.loc[duplicated.treatment == arm, column]
                self.assertAlmostEqual(
                    feature_stats[f"mean_{arm}"], values.mean(), places=9
                )
                self.assertAlmostEqual(
                    feature_stats[f"std_{arm}"], values.std(ddof=1), places=9
                )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_sample_larger_than_input_retains_every_original_row —— 确认目标样本量超过输入时保留全部原始记录。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认目标样本量超过输入时保留全部原始记录。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_sample_larger_than_input_retains_every_original_row(self):
        scanned = scan_raw_data(self.source, sample_size=1000, chunk_size=37)
        sample = scanned["sample"]
        np.testing.assert_array_equal(sample.row_id, np.arange(len(self.frame)))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_readonly_guard_rejects_direct_and_symlinked_data_destinations —— 确认只读保护拒绝直接或通过链接写入数据目录。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认只读保护拒绝直接或通过链接写入数据目录。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_readonly_guard_rejects_direct_and_symlinked_data_destinations(self):
        protected = self.root / "protected_data"
        protected.mkdir()
        marker = protected / "existing.json"
        marker.write_text("original", encoding="utf-8")
        link = self.root / "alias"
        link.symlink_to(protected, target_is_directory=True)
        with patch.object(pipeline_io, "DATA_ROOT", protected):
            for destination in (
                protected,
                marker,
                link / "existing.json",
                protected / "new" / "out.json",
            ):
                with self.subTest(destination=destination), self.assertRaises(
                    ValueError
                ):
                    pipeline_io.write_json({"replace": True}, destination)
            with self.assertRaises(ValueError):
                pipeline_io.save_sample(pd.DataFrame(), link / "sample.npz")
            for destination in (protected, link):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    pipeline_io.write_split_artifacts({}, destination)
            destination = self.root / "artifacts" / "allowed.json"
            pipeline_io.write_json({"ok": True}, destination)
            self.assertTrue(destination.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "original")
        self.assertFalse((protected / "new").exists())
        self.assertFalse((protected / "sample.npz").exists())

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_sample_round_trip_uses_an_output_directory —— 核对建模样本保存和读取后的数据一致性。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对建模样本保存和读取后的数据一致性。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_sample_round_trip_uses_an_output_directory(self):
        scanned = scan_raw_data(
            self.source, sample_size=33, seed=2, chunk_size=41
        )
        sample = scanned["sample"]
        destination = self.root / "artifacts" / "sample.npz"
        pipeline_io.save_sample(sample, destination)
        loaded = pipeline_io.load_sample(destination)
        pd.testing.assert_frame_equal(sample, loaded)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_split_is_disjoint_reproducible_and_preserves_joint_strata —— 确认划分互斥、可复现并保持联合分层比例。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认划分互斥、可复现并保持联合分层比例。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_split_is_disjoint_reproducible_and_preserves_joint_strata(self):
        frame = _synthetic_frame(2000)
        frame.insert(0, "row_id", np.arange(len(frame)))
        parts = data.split_sample(frame, outcome="conversion", seed=42)
        repeated = data.split_sample(frame, outcome="conversion", seed=42)
        expected_sizes = {"train": 1200, "validation": 400, "test": 400}
        identifiers = []
        expected_proportions = frame.groupby(["treatment", "conversion"]).size() / len(
            frame
        )
        for name, part in parts.items():
            self.assertEqual(len(part), expected_sizes[name])
            pd.testing.assert_frame_equal(part, repeated[name])
            proportions = part.groupby(["treatment", "conversion"]).size() / len(part)
            np.testing.assert_allclose(
                proportions, expected_proportions, atol=1.0 / len(part)
            )
            identifiers.append(set(part.row_id))
        self.assertFalse(identifiers[0] & identifiers[1])
        self.assertFalse(identifiers[0] & identifiers[2])
        self.assertFalse(identifiers[1] & identifiers[2])
        self.assertEqual(set.union(*identifiers), set(frame.row_id))
        visit_parts = data.split_sample(frame, outcome="visit", seed=42)
        self.assertEqual(sum(map(len, visit_parts.values())), len(frame))

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_schema_nonfinite_and_nonbinary_labels_fail —— 确认错误列结构、非有限特征及非二分类标签报错。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认错误列结构、非有限特征及非二分类标签报错。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_schema_nonfinite_and_nonbinary_labels_fail(self):
        invalid_frames = []
        invalid_frames.append(self.frame.drop(columns="f11"))
        invalid_frames.append(self.frame.assign(extra=0))
        invalid_frames.append(self.frame[self.frame.columns[::-1]])
        nonfinite = self.frame.copy()
        nonfinite.loc[350, "f0"] = np.inf
        invalid_frames.append(nonfinite)
        missing = self.frame.copy()
        missing.loc[350, "f2"] = np.nan
        invalid_frames.append(missing)
        for label in data.LABELS:
            invalid = self.frame.copy()
            invalid.loc[350, label] = 2
            invalid_frames.append(invalid)
        invalid_frames.append(self.frame.assign(treatment=1))
        for index, frame in enumerate(invalid_frames):
            with self.subTest(invalid_case=index):
                frame.to_csv(self.source, index=False)
                # 即使非法记录不在目标样本中，全量扫描仍应报错。
                with self.assertRaises(ValueError):
                    scan_raw_data(self.source, sample_size=5, chunk_size=73)
        with self.assertRaises(ValueError):
            scan_raw_data(self.source, sample_size=-1)
        with self.assertRaises(ValueError):
            scan_raw_data(self.source, chunk_size=0)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_split_rejects_unsupported_outcome_and_too_small_strata —— 确认划分拒绝不支持的结果列及过小的分层。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认划分拒绝不支持的结果列及过小的分层。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_split_rejects_unsupported_outcome_and_too_small_strata(self):
        with self.assertRaises(ValueError):
            data.split_sample(self.frame, outcome="exposure")
        tiny = self.frame.iloc[:4]
        with self.assertRaises(ValueError):
            data.split_sample(tiny, outcome="conversion")


if __name__ == "__main__":
    unittest.main()
