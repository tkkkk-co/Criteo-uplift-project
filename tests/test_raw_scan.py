from criteo_uplift.config import COLUMNS, FEATURES, LABELS
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from criteo_uplift import diagnostics, preparation, raw_scan
from criteo_uplift.diagnostics import covariate_balance, run_diagnostics


#-----------------------------------------------------------------------------------------------------------------------
# 类：UnifiedScanTests —— 验证统一扫描的统计与抽样。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证统一扫描的统计与抽样。
#-----------------------------------------------------------------------------------------------------------------------
class UnifiedScanTests(unittest.TestCase):
    def setUp(self):
        #-------------
        # 函数名：UnifiedScanTests.setUp
        # 功能：创建临时原始 CSV，包含指定处理比例、近常量特征和重复行，用于统一扫描测试。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；初始化或清理测试实例状态。
        #-------------
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'raw.csv'
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(rng.normal(size=(160, 12)), columns=FEATURES)
        for label in LABELS:
            frame[label] = rng.integers(0, 2, len(frame))
        frame['treatment'] = np.r_[np.zeros(40), np.ones(120)]
        frame['f0'] = 1 + np.arange(len(frame)) * 1e-10
        frame['f1'] = 7.0
        frame.iloc[-1] = frame.iloc[-2]  # 重复行仍参与审计、统计和抽样。
        frame.to_csv(self.path, index=False)
        self.raw = pd.read_csv(self.path, dtype=np.float64)

    def assert_audit_close(self, actual, expected):
        #-------------
        # 保留跨平台浮点容差。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据目录。
        #     - actual：待核验的全量审计字典。
        #     - expected：预期的全量审计字典。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        self.assertEqual({k: v for k, v in actual.items() if k != 'feature_stats'},
                         {k: v for k, v in expected.items() if k != 'feature_stats'})
        pd.testing.assert_frame_equal(pd.DataFrame(actual['feature_stats']),
                                      pd.DataFrame(expected['feature_stats']),
                                      rtol=2e-6, atol=1e-15)

    def scan(self, **kwargs):
        #-------------
        # 函数名：UnifiedScanTests.scan
        # 功能：使用统一测试默认参数调用 scan_raw_data，并允许单项参数覆盖。
        # 输入参数：
        #     - **kwargs：需要覆盖默认扫描设置的关键字参数。
        # 输出参数：
        #     - 一次流式扫描返回的样本、审计和诊断结果字典。
        #-------------
        options = dict(sample_size=43, per_arm=10, seed=31, chunk_size=17, progress=lambda _: None)
        options.update(kwargs)
        return raw_scan.scan_raw_data(self.path, **options)

    def test_one_csv_pass_preserves_both_sampling_streams_and_full_statistics(self):
        #-------------
        # 函数名：UnifiedScanTests.test_one_csv_pass_preserves_both_sampling_streams_and_full_statistics
        # 功能：验证单次 CSV 扫描同时保留建模抽样、C2ST 两臂抽样和全量统计结果。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        before = self.path.read_bytes()
        with patch.object(diagnostics.pd, 'read_csv', wraps=pd.read_csv) as read:
            scanned = self.scan()
        read.assert_called_once()
        self.assertIn('chunksize', read.call_args.kwargs)
        self.assertEqual(scanned['source']['sha256'], hashlib.sha256(before).hexdigest())

        # 直接为所有原始行生成优先级，验证统一扫描仍选中了原来的样本。
        keys = np.random.default_rng(31).random(len(self.raw))
        expected_ids = np.sort(np.argsort(keys)[:43])
        np.testing.assert_array_equal(scanned['sample'].row_id, expected_ids)
        diagnostic_ids = []
        for arm, child in enumerate(np.random.SeedSequence(31).spawn(2)):
            ids = np.flatnonzero(self.raw.treatment.to_numpy() == arm)
            keys = np.random.default_rng(child).random(len(ids))
            diagnostic_ids.extend(ids[np.argsort(keys)[:10]])
        np.testing.assert_array_equal(scanned['c2st_sample'].row_id, sorted(diagnostic_ids))
        sample = scanned['c2st_sample']
        np.testing.assert_array_equal(sample[FEATURES], self.raw.iloc[sample.row_id][FEATURES])
        self.assertEqual(sample.f0.dtype, np.dtype('float64'))
        self.assertEqual(scanned['sample'].f0.dtype, np.dtype('float32'))

        expected = covariate_balance(self.raw, FEATURES)
        pd.testing.assert_frame_equal(scanned['balance'][expected.columns], expected)
        audit = scanned['audit']
        self.assertEqual(audit['duplicate_row_fingerprints'], 1)
        self.assertEqual(audit['n_rows'], len(self.raw))
        for record in audit['feature_stats']:
            for arm in (0, 1):
                values = self.raw.loc[self.raw.treatment == arm, record['feature']]
                self.assertEqual(record[f'mean_{arm}'], values.mean())
                self.assertEqual(record[f'std_{arm}'], values.std(ddof=1))
            self.assertEqual(record['smd'], expected.set_index('feature').loc[record['feature'], 'smd'])
        self.assertEqual(scanned['srm']['observed_ratio'], 0.75)
        self.assertEqual(sample.treatment.mean(), 0.5)
        self.assertEqual(self.path.read_bytes(), before)

    def test_diagnostic_sample_independent_of_model_size_and_chunk_size(self):
        #-------------
        # 函数名：UnifiedScanTests.test_diagnostic_sample_independent_of_model_size_and_chunk_size
        # 功能：验证 C2ST 诊断抽样不受建模样本规模和 CSV 分块大小影响。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        baseline = self.scan()
        for size in (1, 0, 200):
            with self.subTest(sample_size=size):
                scanned = self.scan(sample_size=size, chunk_size=53)
                pd.testing.assert_frame_equal(scanned['c2st_sample'], baseline['c2st_sample'])
                pd.testing.assert_frame_equal(scanned['balance'], baseline['balance'])
                if size is None:
                    self.assertIsNone(scanned['sample'])
                    self.assertIsNone(scanned['sample_report'])
                    self.assertEqual(scanned['audit']['n_rows'], len(self.raw))
                    self.assertNotIn('sample_n', scanned['audit'])
                else:
                    self.assertEqual(len(scanned["sample"]), min(size or len(self.raw), len(self.raw)))
                    self.assertEqual(scanned['sample_report']['sample_size'], size)
                    self.assertEqual(scanned['sample_report']['sample_n'], len(scanned['sample']))
                    self.assertEqual(
                        scanned['sample_report']['sampling'],
                        'all rows' if size == 0 else 'uniform without replacement, random-priority reservoir',
                    )

    def test_source_changes_and_insufficient_arms_fail(self):
        #-------------
        # 函数名：UnifiedScanTests.test_source_changes_and_insufficient_arms_fail
        # 功能：验证扫描期间原始文件变化或某处理臂样本不足时会明确失败。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        with patch.object(raw_scan, '_file_identity', side_effect=[(1,), (2,)]):
            with self.assertRaisesRegex(RuntimeError, '发生变化'):
                self.scan()
        with self.assertRaisesRegex(ValueError, '每个处理臂至少需要 41 行'):
            self.scan(per_arm=41)

    def test_update_reuses_scan_and_rejects_mismatched_settings(self):
        #-------------
        # 函数名：UnifiedScanTests.test_update_reuses_scan_and_rejects_mismatched_settings
        # 功能：验证诊断复用已有扫描结果，并拒绝与原始扫描配置不匹配的设置。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        scanned = self.scan()
        out = self.root / 'diagnostics'
        out.mkdir()
        with patch('criteo_uplift.preparation.scan_raw_data', side_effect=AssertionError('second scan')):
            result = run_diagnostics(
                out, train=self.raw, validation=self.raw, audit=scanned['audit'], source=scanned['source'],
                sample_size=20, permutations=1, seed=31, scanned=scanned, progress=lambda _: None,
            )
            self.assertEqual(result['full_n'], len(self.raw))
            for seed, size in ((32, 20), (31, 40)):
                with self.assertRaisesRegex(ValueError, '规模或种子'):
                    run_diagnostics(
                        out, train=self.raw, validation=self.raw, audit=scanned['audit'], source=scanned['source'],
                        sample_size=size, permutations=1, seed=seed, scanned=scanned,
                    )
        self.assertEqual(json.loads((out / 'srm.json').read_text())['observed_ratio'], .75)


    def test_every_run_scans_source_without_touching_stale_artifacts(self):
        stale = self.root / 'artifacts/cache'
        stale.mkdir(parents=True)
        names = ('sample_43_seed31.npz', 'sample_43_seed31.json',
                 'full_audit.json', 'c2st_sample_n20_seed31.npz')
        for name in names:
            (stale / name).write_text('stale', encoding='utf-8')
        args = SimpleNamespace(data=self.path, sample_size=43, c2st_sample=20, seed=31)
        with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan:
            for _ in range(2):
                result = preparation.prepare_sample(args, progress=lambda _: None, chunk_size=17)
                self.assertEqual(len(result['sample']), 43)
                self.assertEqual(result['audit']['n_rows'], len(self.raw))
                self.assertEqual(len(result['c2st_sample']), 20)
        self.assertEqual(scan.call_count, 2)
        self.assertEqual({path.name: path.read_text() for path in stale.iterdir()},
                         dict.fromkeys(names, 'stale'))

    def test_duplicate_excess_counts_are_not_unique_duplicate_patterns(self):
        #-------------
        # 函数名：UnifiedScanTests.test_duplicate_excess_counts_are_not_unique_duplicate_patterns
        # 功能：验证重复行多余计数采用总行数减不同指纹数，而不是重复模式种数。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        frame = self.raw.copy()
        frame.iloc[-3:] = frame.iloc[0].to_numpy()
        frame.iloc[-5:-3] = frame.iloc[1].to_numpy()
        frame.to_csv(self.path, index=False)
        audit = self.scan(chunk_size=7)['audit']
        expected = int(frame.duplicated().sum())
        unique_patterns = len(frame[frame.duplicated(keep=False)].drop_duplicates())
        self.assertEqual(expected, 5)
        self.assertEqual(unique_patterns, 2)
        self.assertEqual(audit['duplicate_row_fingerprints'], expected)
        self.assertEqual(audit['duplicate_fraction_denominator'], len(frame))
        self.assertEqual(audit['duplicate_row_fraction'], expected / len(frame))
        self.assertIn('k-1', audit['duplicate_count_definition'])
        self.assertIn('碰撞', audit['duplicate_method'])
        self.assertIn('index=False', audit['duplicate_method'])


    def test_fixed_reference_and_c2st_results(self):
        #-------------
        # 与合成数据基准统计和行号逐项对照，并验证实际 C2ST 检验输出一致。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        reference = json.loads((Path(__file__).parent / 'fixtures/raw_scan_reference.json').read_text())
        for chunk_size in (1, 17, 53, 160):
            with self.subTest(chunk_size=chunk_size):
                scanned = self.scan(chunk_size=chunk_size)
                self.assertEqual(scanned['sample'].row_id.tolist(), reference['sample_ids'])
                self.assertEqual(scanned['c2st_sample'].row_id.tolist(), reference['c2st_ids'])
                self.assert_audit_close(scanned['audit'], reference['audit'])
                # 均值、SMD 和 KS P-value 分别使用严格浮点容差；KS Statistic、人数和说明仍精确比较。
                balance = scanned['balance'].drop(columns='source_sha256')
                expected_balance = pd.DataFrame(reference['balance'])
                moments = ['mean_treatment', 'mean_control', 'smd', 'abs_smd']
                pd.testing.assert_frame_equal(balance[moments], expected_balance[moments],
                                              check_exact=False, rtol=1e-12, atol=1e-15)
                pd.testing.assert_series_equal(balance['ks_pvalue'], expected_balance['ks_pvalue'],
                                               check_exact=False, rtol=1e-12, atol=1e-15)
                exact_columns = balance.columns.difference(moments + ['ks_pvalue'])
                pd.testing.assert_frame_equal(balance[exact_columns],
                                              expected_balance[exact_columns], check_exact=True)
                # SRM P-value 在 Linux 与 macOS 的尾数存在舍入差异，其他 SRM 字段仍精确比较。
                np.testing.assert_allclose(scanned['srm']['pvalue'], reference['srm']['pvalue'],
                                           rtol=1e-12, atol=1e-15)
                self.assertEqual(
                    {k: v for k, v in scanned['srm'].items() if k not in ('source_sha256', 'pvalue')},
                    {k: v for k, v in reference['srm'].items() if k != 'pvalue'},
                )
        # 从原始 float64 表恢复相同输入，独立运行分类器。
        expected_sample = self.raw.iloc[reference['c2st_ids']]
        expected = diagnostics.c2st(expected_sample, FEATURES, sample_size=20, permutations=2, seed=31)
        actual = diagnostics.c2st(scanned['c2st_sample'], FEATURES, sample_size=20, permutations=2, seed=31)
        for key in ('summary', 'null_auc'):
            pd.testing.assert_frame_equal(actual[key], expected[key], check_exact=True)

    def test_scan_resources_close_on_invalid_late_chunk(self):
        #-------------
        # 末尾非法行仍会失败，并关闭读取器和全部诊断临时文件，不改动来源。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        from tempfile import TemporaryFile
        files, streams = [], []
        original_read = pd.read_csv
        def temporary_file():
            #-------------
            # 创建诊断临时文件并登记，以便检查失败路径是否关闭资源。
            # 输入参数：
            #     - 无。
            # 输出参数：
            #     - 新建的临时二进制文件对象，同时加入测试文件列表。
            #-------------
            file = TemporaryFile()
            files.append(file)
            return file
        def read_csv(*args, **kwargs):
            #-------------
            # 调用原 CSV 读取函数并登记分块读取器，以便检查关闭状态。
            # 输入参数：
            #     - *args：转交 pandas.read_csv 的位置参数，通常包含原始文件路径。
            #     - **kwargs：转交 pandas.read_csv 的关键字参数，包括分块大小和 dtype。
            # 输出参数：
            #     - pandas.read_csv 返回的 DataFrame 或 TextFileReader。
            #-------------
            result = original_read(*args, **kwargs)
            if 'chunksize' in kwargs:
                streams.append(args[0])
            return result
        frame = self.raw.copy()
        frame.loc[len(frame) - 1, 'conversion'] = 2
        frame.to_csv(self.path, index=False)
        before = self.path.read_bytes()
        with patch.object(raw_scan, 'TemporaryFile', side_effect=temporary_file), \
             patch.object(raw_scan.pd, 'read_csv', side_effect=read_csv):
            with self.assertRaises(ValueError):
                self.scan()
        self.assertEqual(len(files), len(FEATURES) + 1)
        self.assertTrue(all(file.closed for file in files))
        self.assertEqual(len(streams), 1)
        self.assertTrue(streams[0].closed)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
