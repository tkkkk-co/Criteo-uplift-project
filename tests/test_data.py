from criteo_uplift.config import COLUMNS, FEATURES, LABELS
# 项目测试：验证数据读取、抽样和划分。

from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from criteo_uplift import pipeline_io, preparation
from criteo_uplift.raw_scan import scan_raw_data


def _synthetic_frame(n=400):
    #-------------
    # 函数名：_synthetic_frame
    # 功能：生成包含 f0–f11、treatment、conversion、visit 和 exposure 的合成测试数据。
    # 输入参数：
    #     - n：需要生成的合成数据行数。
    # 输出参数：
    #     - 合成数据 DataFrame。
    #-------------
    row = np.arange(n)
    frame = pd.DataFrame(
        {feature: row * (j + 1.0) + j / 8.0 for j, feature in enumerate(FEATURES)}
    )
    frame["treatment"] = row % 2
    frame["conversion"] = (row // 2) % 2
    frame["visit"] = np.maximum(frame.conversion, row % 5 == 0).astype(int)
    frame["exposure"] = frame.treatment * (row % 3 != 0)
    return frame[COLUMNS]


#-----------------------------------------------------------------------------------------------------------------------
# 类：DataTests —— 验证数据读取、抽样和划分。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证数据读取、抽样和划分。
#-----------------------------------------------------------------------------------------------------------------------
class DataTests(unittest.TestCase):
    def setUp(self):
        #-------------
        # 函数名：DataTests.setUp
        # 功能：创建数据准备测试使用的临时项目目录和合成 CSV 文件。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化测试实例的临时路径和数据属性。
        #-------------
        self.directory = tempfile.TemporaryDirectory(prefix="criteo_data_test_")
        self.root = Path(self.directory.name)
        self.source = self.root / "synthetic.csv"
        self.frame = _synthetic_frame()
        self.frame.to_csv(self.source, index=False)

    def tearDown(self):
        #-------------
        # 函数名：DataTests.tearDown
        # 功能：清理数据准备测试创建的临时目录。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；释放测试资源。
        #-------------
        self.directory.cleanup()

    def test_sampling_reads_entire_file_is_reproducible_and_chunk_independent(self):
        #-------------
        # 函数名：DataTests.test_sampling_reads_entire_file_is_reproducible_and_chunk_independent
        # 功能：验证分块扫描读取完整 CSV，抽样结果可复现且不受 chunk_size 影响。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        before = preparation.fingerprint(self.source)
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
        self.assertNotIn("sample_n", audit)
        self.assertEqual(audit["n_treatment"], 200)
        self.assertEqual(audit["groups"]["0"]["conversion"]["positive"], 100)
        self.assertEqual(preparation.fingerprint(self.source), before)

    def test_full_data_audit_counts_duplicates_without_removing_rows(self):
        #-------------
        # 函数名：DataTests.test_full_data_audit_counts_duplicates_without_removing_rows
        # 功能：验证全量审计能够统计重复行，但不会自动删除原始记录。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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

    def test_sample_larger_than_input_retains_every_original_row(self):
        #-------------
        # 函数名：DataTests.test_sample_larger_than_input_retains_every_original_row
        # 功能：验证目标抽样量大于原始数据行数时，样本保留全部原始行。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        scanned = scan_raw_data(self.source, sample_size=1000, chunk_size=37)
        sample = scanned["sample"]
        np.testing.assert_array_equal(sample.row_id, np.arange(len(self.frame)))

    def test_readonly_guard_rejects_direct_and_symlinked_data_destinations(self):
        #-------------
        # 函数名：DataTests.test_readonly_guard_rejects_direct_and_symlinked_data_destinations
        # 功能：验证输出路径保护同时拒绝直接写入 data 目录和通过符号链接写入。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
                    pipeline_io.write_result({"replace": True}, destination)
            with self.assertRaises(ValueError):
                pipeline_io.save_sample(pd.DataFrame(), link / "sample.npz")
            for destination in (protected, link):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    pipeline_io.write_result(pd.DataFrame({"n": [1]}), destination / "summary.csv")
            destination = self.root / "artifacts" / "allowed.json"
            pipeline_io.write_result({"ok": True}, destination)
            self.assertTrue(destination.is_file())
        self.assertEqual(marker.read_text(encoding="utf-8"), "original")
        self.assertFalse((protected / "new").exists())
        self.assertFalse((protected / "sample.npz").exists())

    def test_sample_round_trip_uses_an_output_directory(self):
        #-------------
        # 函数名：DataTests.test_sample_round_trip_uses_an_output_directory
        # 功能：验证样本缓存保存和读取往返一致，且输出路径位于允许的结果目录。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        scanned = scan_raw_data(
            self.source, sample_size=33, seed=2, chunk_size=41
        )
        sample = scanned["sample"]
        destination = self.root / "artifacts" / "sample.npz"
        pipeline_io.save_sample(sample, destination)
        loaded = pipeline_io.load_sample(destination)
        pd.testing.assert_frame_equal(sample, loaded)

    def test_split_is_disjoint_reproducible_and_preserves_joint_strata(self):
        #-------------
        # 函数名：DataTests.test_split_is_disjoint_reproducible_and_preserves_joint_strata
        # 功能：验证 Train/Validation/Test 划分互斥、可复现，并保留 T×Y 联合分层。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = _synthetic_frame(2000)
        frame.insert(0, "row_id", np.arange(len(frame)) * 3 + 100)
        parts, summary, overlap = preparation.split_sample(frame, outcome="conversion", seed=42)
        repeated, repeated_summary, repeated_overlap = preparation.split_sample(frame, outcome="conversion", seed=42)
        pd.testing.assert_frame_equal(summary, repeated_summary)
        self.assertEqual(overlap, repeated_overlap)
        self.assertTrue(all(overlap['row_id_checks'].values()))
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
        visit_parts, visit_summary, _ = preparation.split_sample(frame, outcome="visit", seed=42)
        self.assertEqual(sum(map(len, visit_parts.values())), len(frame))
        for splits, table in ((parts, summary), (visit_parts, visit_summary)):
            self.assertEqual(list(table.columns), [
                "split", "treatment", "n", "conversion_positive", "visit_positive",
            ])
            self.assertEqual(len(table), 6)
            self.assertEqual(table.n.sum(), len(frame))
            for name, part in splits.items():
                for arm in (0, 1):
                    rows = part[part.treatment == arm]
                    actual = table[(table.split == name) & (table.treatment == arm)].iloc[0]
                    self.assertEqual(actual.n, len(rows))
                    self.assertEqual(actual.conversion_positive, int(rows.conversion.sum()))
                    self.assertEqual(actual.visit_positive, int(rows.visit.sum()))

        _, conversion_only, _ = preparation.split_sample(frame.drop(columns="visit"), seed=42)
        self.assertNotIn("visit_positive", conversion_only.columns)
        self.assertEqual(conversion_only.conversion_positive.sum(), frame.conversion.sum())

    def test_split_rejects_missing_null_or_duplicate_source_ids(self):
        #-------------
        # 函数名：DataTests.test_split_rejects_missing_null_or_duplicate_source_ids
        # 功能：验证样本划分拒绝缺失、空值或重复的来源 row_id。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = _synthetic_frame(2000)
        frame.insert(0, "row_id", np.arange(len(frame)))
        missing = frame.drop(columns="row_id")
        null = frame.copy()
        null.loc[0, "row_id"] = np.nan
        duplicated = frame.copy()
        duplicated.loc[1, "row_id"] = duplicated.loc[0, "row_id"]
        for invalid in (missing, null, duplicated):
            with self.subTest(columns=invalid.columns.tolist()), self.assertRaisesRegex(ValueError, "row_id"):
                preparation.split_sample(invalid)

    def test_split_detects_corrupted_partition_membership(self):
        #-------------
        # 函数名：DataTests.test_split_detects_corrupted_partition_membership
        # 功能：验证划分完整性检查能够发现集合成员被篡改或遗漏。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = _synthetic_frame(2000)
        frame.insert(0, "row_id", np.arange(len(frame)))
        train, holdout = np.arange(1200), np.arange(1200, 2000)
        validation, test = np.arange(1200, 1600), np.arange(1600, 2000)
        cases = [
            (np.r_[train[:-1], train[0]], validation, test, "train.*唯一"),
            (train, np.r_[train[0], validation[1:]], test, "train 与 validation"),
            (train, validation, np.r_[train[0], test[1:]], "train 与 test"),
            (train, validation, np.r_[validation[0], test[1:]], "validation 与 test"),
            (train, validation, test[:-1], "完整覆盖"),
        ]
        for train_ids, validation_ids, test_ids, error in cases:
            with self.subTest(error=error), patch.object(
                preparation, "train_test_split",
                side_effect=[(train_ids, holdout), (validation_ids, test_ids)],
            ), self.assertRaisesRegex(ValueError, error):
                preparation.split_sample(frame)

    def test_split_reports_all_feature_pairs_without_treating_matches_as_row_duplicates(self):
        #-------------
        # 函数名：DataTests.test_split_reports_all_feature_pairs_without_treating_matches_as_row_duplicates
        # 功能：验证相同匿名特征向量会被报告为特征重叠，而不会误判为来源行重复。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = _synthetic_frame(15)
        frame.insert(0, 'row_id', np.arange(15))
        frame[FEATURES] = frame[FEATURES].astype('float32')
        # 不同 row_id 保持原样，只复制特征；重复训练向量不应重复计数。
        frame.loc[[1, 9, 10, 12], FEATURES] = frame.loc[0, FEATURES].to_numpy()
        frame.loc[13, FEATURES] = frame.loc[11, FEATURES].to_numpy()
        with patch.object(preparation, 'train_test_split', side_effect=[
            (np.arange(9), np.arange(9, 15)),
            (np.arange(9, 12), np.arange(12, 15)),
        ]):
            parts, _, overlap = preparation.split_sample(frame)
        self.assertTrue(all(overlap['row_id_checks'].values()))
        self.assertEqual(overlap['fraction_matching_training_features'], {
            'validation': 2 / 3, 'test': 1 / 3,
        })
        self.assertEqual(overlap['fraction_test_matching_validation_features'], 2 / 3)
        self.assertEqual(overlap['features'], FEATURES)
        self.assertEqual(set(overlap['feature_dtypes'].values()), {'float32'})
        self.assertEqual(sum(map(len, parts.values())), len(frame))

    def test_invalid_schema_nonfinite_and_nonbinary_labels_fail(self):
        #-------------
        # 函数名：DataTests.test_invalid_schema_nonfinite_and_nonbinary_labels_fail
        # 功能：验证 CSV Schema 错误、非有限特征和非二元标签会触发明确异常。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
        for label in LABELS:
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

    def test_split_rejects_unsupported_outcome_and_too_small_strata(self):
        #-------------
        # 函数名：DataTests.test_split_rejects_unsupported_outcome_and_too_small_strata
        # 功能：验证划分拒绝不支持的结果变量及样本量不足的联合分层。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with self.assertRaises(ValueError):
            preparation.split_sample(self.frame, outcome="exposure")
        tiny = self.frame.iloc[:4].assign(row_id=np.arange(4))
        with self.assertRaises(ValueError):
            preparation.split_sample(tiny, outcome="conversion")


    def test_write_result_preserves_json_values_and_csv_columns(self):
        #-------------
        # 函数名：DataTests.test_write_result_preserves_json_values_and_csv_columns
        # 功能：验证 JSON/CSV 写入保留数值内容和 DataFrame 列结构。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        report = {"中文": np.int64(3), "array": np.array([1.5, np.nan]),
                  "path": Path("result"), "infinity": float("inf")}
        path = self.root / "results" / "audit.json"
        pipeline_io.write_result(report, path)
        self.assertEqual(json.loads(path.read_text()), {
            "中文": 3, "array": [1.5, None], "path": "result", "infinity": None,
        })
        frame = pd.DataFrame({"split": ["train", "test"], "n": [8, 2]}, index=[10, 20])
        target = path.with_name("split_summary.csv")
        pipeline_io.write_result(frame, target)
        pd.testing.assert_frame_equal(pd.read_csv(target), frame.reset_index(drop=True))

    def test_write_result_rejects_invalid_format_before_writing(self):
        #-------------
        # 函数名：DataTests.test_write_result_rejects_invalid_format_before_writing
        # 功能：验证不支持的结果文件格式会在写入前被拒绝。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for value, name, error in [
            ({"n": 1}, "summary.csv", TypeError),
            (pd.DataFrame({"n": [1]}), "audit.json", TypeError),
            ({"n": 1}, "report.txt", ValueError),
        ]:
            target = self.root / "invalid" / name
            with self.subTest(name=name), self.assertRaises(error):
                pipeline_io.write_result(value, target)
            self.assertFalse(target.parent.exists())
        target = self.root / "existing.csv"
        target.write_text("original")
        with self.assertRaises(TypeError):
            pipeline_io.write_result({"n": 1}, target)
        self.assertEqual(target.read_text(), "original")


if __name__ == "__main__":
    unittest.main()


class InputChecksTests(unittest.TestCase):
    def test_checks_do_not_transform_inputs(self):
        #-------------
        # 函数名：InputChecksTests.test_checks_do_not_transform_inputs
        # 功能：验证公共输入检查只校验数据，不对传入数组或 DataFrame 执行原地修改。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        from criteo_uplift.input_checks import check_binary_vector, check_feature_matrix, check_row_ids
        x = np.array([[1., np.nan], [2., 3.]])
        before = x.copy()
        self.assertIsNone(check_feature_matrix(x, allow_nan=True))
        np.testing.assert_array_equal(x, before)
        with self.assertRaises(ValueError): check_feature_matrix(x)
        with self.assertRaises(ValueError): check_feature_matrix([[np.inf]], allow_nan=True)
        with self.assertRaises(ValueError): check_feature_matrix([["a"]])
        with self.assertRaises(ValueError): check_feature_matrix(np.ones((2, 2)), n_features=3)
        for values in ([], [[0, 1]], [0, 2], [0, np.nan]):
            with self.assertRaises(ValueError): check_binary_vector(values)
        labels = np.array([False, True]); check_binary_vector(labels, n_rows=2, require_both=True)
        self.assertEqual(labels.dtype, np.dtype(bool))
        for ids in ([], [1, 1], [1, None], [[1, 2]]):
            with self.assertRaises(ValueError): check_row_ids(ids)
        with self.assertRaises(ValueError): check_row_ids([-1, 2], require_integer=True)
        check_row_ids(["a", "b"], n_rows=2)

    def test_corrupt_cached_arrays_are_rejected(self):
        #-------------
        # 函数名：InputChecksTests.test_corrupt_cached_arrays_are_rejected
        # 功能：验证形状、标签或来源行号损坏的 NPZ 缓存无法被静默加载。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'corrupt.npz'
            valid = {'X': np.ones((3, 12)), 'labels': np.zeros((3, 4), dtype=np.int8),
                     'row_id': np.arange(3)}
            for changed in ({'row_id': np.array([0, 0, 1])},
                            {'X': np.ones((3, 11))},
                            {'X': np.full((3, 12), np.inf)},
                            {'labels': np.full((3, 4), 2)},
                            {'labels': np.zeros((2, 4))}):
                np.savez(path, **{**valid, **changed})
                with self.assertRaises(ValueError):
                    pipeline_io.load_sample(path)
