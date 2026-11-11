from criteo_uplift.config import COLUMNS, FEATURES, LABELS
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from itertools import product
import json
import unittest
from zipfile import ZipFile, ZIP_DEFLATED
from zlib import error as ZlibError

import numpy as np
import pandas as pd

from criteo_uplift import diagnostics, preparation, raw_scan
from criteo_uplift.diagnostics import covariate_balance, run_diagnostics
from criteo_uplift.pipeline_io import load_sample


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
        # 全列与流式统计的浮点归约顺序不同，沿用近常量特征已有的误差容限。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
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
        self.assertEqual(sum('chunksize' in c.kwargs for c in read.call_args_list), 1)

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

    def test_audit_only_variance_matches_full_diagnostics_for_near_constant_features(self):
        #-------------
        # 函数名：UnifiedScanTests.test_audit_only_variance_matches_full_diagnostics_for_near_constant_features
        # 功能：验证关闭完整诊断时，流式累计的近常量特征方差与完整诊断结果一致。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        expected = self.scan()['audit']['feature_stats']
        for chunk_size in (13, 31, len(self.raw)):
            actual = self.scan(diagnostics_enabled=False, chunk_size=chunk_size)['audit']['feature_stats']
            for observed, reference in zip(actual, expected):
                with self.subTest(chunk_size=chunk_size, feature=reference['feature']):
                    for field in ('mean_0', 'mean_1', 'std_0', 'std_1', 'smd'):
                        np.testing.assert_allclose(observed[field], reference[field], rtol=2e-6, atol=1e-15)

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
        for size in (None, 0, 200):
            with self.subTest(sample_size=size):
                scanned = self.scan(sample_enabled=size is not None, sample_size=size, chunk_size=53)
                pd.testing.assert_frame_equal(scanned['c2st_sample'], baseline['c2st_sample'])
                pd.testing.assert_frame_equal(scanned['balance'], baseline['balance'])
                if size is None:
                    self.assertIsNone(scanned['sample'])
                    self.assertIsNone(scanned['sample_report'])
                    self.assertEqual(scanned['audit']['n_rows'], len(self.raw))
                    self.assertNotIn('sample_n', scanned['audit'])
                else:
                    self.assertEqual(len(scanned['sample']), len(self.raw))
                    self.assertEqual(scanned['sample_report']['sample_size'], size)
                    self.assertEqual(scanned['sample_report']['sample_n'], len(self.raw))
                    self.assertEqual(
                        scanned['sample_report']['sampling'],
                        'all rows' if size == 0 else 'uniform without replacement, random-priority reservoir',
                    )

    def test_audit_diagnostics_and_two_samples_are_independent(self):
        #-------------
        # 函数名：UnifiedScanTests.test_audit_diagnostics_and_two_samples_are_independent
        # 功能：验证全量审计、平衡诊断、建模样本和 C2ST 样本使用独立的计算/抽样路径。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        expected_balance = covariate_balance(self.raw, FEATURES)
        baseline = self.scan()
        for audit, checks, sample_on in product((False, True), repeat=3):
            options = dict(
                audit_enabled=audit, diagnostics_enabled=checks,
                sample_enabled=sample_on, sample_size=43, per_arm=10,
            )
            with self.subTest(**options):
                if not any((audit, checks, sample_on)):
                    with self.assertRaisesRegex(ValueError, '至少启用'):
                        self.scan(**options)
                    continue
                scanned = self.scan(**options)
                for name in ('sample_enabled', 'audit_enabled', 'diagnostics_enabled'):
                    self.assertEqual(scanned['metadata'][name], options[name])
                self.assertEqual(scanned['metadata']['c2st_sample_enabled'], checks)
                if audit:
                    self.assertEqual(scanned['audit']['n_rows'], len(self.raw))
                    self.assertEqual(scanned['audit']['duplicate_row_fingerprints'], 1)
                    self.assertEqual(scanned['audit']['n_control'], 40)
                    self.assertEqual(len(scanned['audit']['feature_stats']), 12)
                    for group, (name, rows) in zip(scanned['audit']['group_summary'], (
                        ('Control', self.raw[self.raw.treatment == 0]),
                        ('Treatment', self.raw[self.raw.treatment == 1]),
                        ('Total', self.raw),
                    )):
                        self.assertEqual(group['group'], name)
                        self.assertEqual(group['n'], len(rows))
                        self.assertEqual(group['fraction'], len(rows) / len(self.raw))
                        self.assertEqual(group['conversion_positive'], int(rows.conversion.sum()))
                        self.assertAlmostEqual(group['conversion_rate'], float(rows.conversion.mean()))
                    for field in ('sample_n', 'sample_seed', 'sampling'):
                        self.assertNotIn(field, scanned['audit'])
                else:
                    self.assertIsNone(scanned['audit'])
                if checks:
                    pd.testing.assert_frame_equal(scanned['balance'][expected_balance.columns], expected_balance)
                    self.assertEqual(scanned['srm']['n_treatment'], 120)
                    self.assertEqual(scanned['srm']['observed_ratio'], .75)
                else:
                    self.assertIsNone(scanned['balance'])
                    self.assertIsNone(scanned['srm'])
                if not sample_on:
                    self.assertIsNone(scanned['sample'])
                    self.assertIsNone(scanned['sample_report'])
                else:
                    pd.testing.assert_frame_equal(scanned['sample'], baseline['sample'])
                    report = scanned['sample_report']
                    self.assertEqual(report, baseline['sample_report'])
                    self.assertEqual(report['scope'], 'modeling_sample')
                    self.assertEqual(report['source'], scanned['source'])
                    self.assertEqual(report['source_n'], len(self.raw))
                    self.assertEqual(report['sample_n'], len(scanned['sample']))
                    self.assertEqual(report['seed'], 31)
                    self.assertEqual(report['feature_representation'], 'float32')
                    self.assertEqual(report['n_treatment'], int(scanned['sample'].treatment.sum()))
                    self.assertEqual(report['n_control'], int((scanned['sample'].treatment == 0).sum()))
                    for group, (name, rows) in zip(report['group_summary'], (
                        ('Control', scanned['sample'][scanned['sample'].treatment == 0]),
                        ('Treatment', scanned['sample'][scanned['sample'].treatment == 1]),
                        ('Total', scanned['sample']),
                    )):
                        self.assertEqual(group['group'], name)
                        self.assertEqual(group['n'], len(rows))
                        self.assertEqual(group['fraction'], len(rows) / len(scanned['sample']))
                        self.assertEqual(group['conversion_positive'], int(rows.conversion.sum()))
                        self.assertAlmostEqual(group['conversion_rate'], float(rows.conversion.mean()))
                    self.assertNotIn('audit', report)
                if not checks:
                    self.assertIsNone(scanned['c2st_sample'])
                    self.assertIsNone(scanned['metadata']['sample_n'])
                else:
                    pd.testing.assert_frame_equal(scanned['c2st_sample'], baseline['c2st_sample'])

    def test_source_changes_and_insufficient_arms_fail(self):
        #-------------
        # 函数名：UnifiedScanTests.test_source_changes_and_insufficient_arms_fail
        # 功能：验证扫描期间原始文件变化或某处理臂样本不足时会明确失败。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        source = preparation.fingerprint(self.path)
        with patch.object(raw_scan, 'fingerprint', side_effect=[source, {**source, 'sha256': 'changed'}]):
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

    def test_prepare_sample_scans_once_on_miss_and_never_on_full_hit(self):
        #-------------
        # 函数名：UnifiedScanTests.test_prepare_sample_scans_once_on_miss_and_never_on_full_hit
        # 功能：验证新建缓存只解析一次 CSV，全部缓存命中时不再解析 CSV。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        with patch.object(diagnostics.pd, 'read_csv', wraps=pd.read_csv) as read:
            sample, audit, source, prepared = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertEqual(sum('chunksize' in c.kwargs for c in read.call_args_list), 1)
        with patch.object(diagnostics.pd, 'read_csv', wraps=pd.read_csv) as read, patch.object(
            diagnostics.pd.util, 'hash_pandas_object', side_effect=AssertionError('repeated audit'),
        ):
            cached, cached_audit, cached_source, cached_diagnostics = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        read.assert_not_called()
        pd.testing.assert_frame_equal(sample, cached)
        pd.testing.assert_frame_equal(prepared['c2st_sample'], cached_diagnostics['c2st_sample'])
        self.assertEqual(audit['n_rows'], cached_audit['n_rows'])
        self.assertEqual(source, cached_source)
        self.assertNotIn('sample', prepared)  # 不让后续诊断持有百万行建模表。

    def test_prepare_sample_rebuilds_without_metadata_even_with_legacy_files(self):
        #-------------
        # 函数名：UnifiedScanTests.test_prepare_sample_rebuilds_without_metadata_even_with_legacy_files
        # 功能：验证缺少或不完整的 Sample Metadata 时，即使存在旧缓存文件也会重新构建样本。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        args = SimpleNamespace(
            data=self.path, sample_size=1_000_000, seed=42,
            c2st_sample=20, permutations=1,
        )
        sample, audit, source, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        base = self.root / 'artifacts'
        cache = base / 'cache/sample_1000000_seed42.npz'
        metadata = cache.with_suffix('.json')
        info = json.loads(metadata.read_text())
        self.assertEqual(info['source'], source)
        self.assertEqual(info['sample_size'], args.sample_size)
        self.assertEqual(info['seed'], args.seed)
        self.assertNotIn('audit', info)
        self.assertEqual(info['sample_n'], len(sample))
        self.assertEqual(info['source_n'], audit['n_rows'])

        # 模拟说明缺失的旧缓存；即使旧指纹匹配，也必须重新抽样和审计。
        metadata.rename(self.root / 'metadata_backup.json')
        cache.write_bytes(b'old cache must not be loaded')
        legacy_source = base / 'source_fingerprint.json'
        legacy_audit = base / 'full_audit.json'
        legacy_source.write_text(json.dumps(source))
        legacy_audit.write_text('old audit must not be read')
        legacy_bytes = [p.read_bytes() for p in (legacy_source, legacy_audit)]
        with patch('criteo_uplift.pipeline_io.load_sample', side_effect=AssertionError('old cache reused')):
            rebuilt, rebuilt_audit, rebuilt_source, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)

        pd.testing.assert_frame_equal(rebuilt, sample)
        pd.testing.assert_frame_equal(load_sample(cache), sample)
        self.assertEqual(rebuilt_audit, audit)
        self.assertEqual(rebuilt_source, source)
        self.assertEqual(json.loads(metadata.read_text()), info)
        self.assertEqual([p.read_bytes() for p in (legacy_source, legacy_audit)], legacy_bytes)


    def test_sample_and_audit_caches_are_reused_independently(self):
        #-------------
        # 函数名：UnifiedScanTests.test_sample_and_audit_caches_are_reused_independently
        # 功能：验证 Sample Cache 与 Audit Cache 可独立命中、失效和重新生成。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        for reuse_sample, reuse_audit in product((False, True), repeat=2):
            with self.subTest(reuse_sample=reuse_sample, reuse_audit=reuse_audit):
                root = self.root / f'combination_{reuse_sample}_{reuse_audit}'
                root.mkdir()
                args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
                _, expected_audit, _, _ = preparation.prepare_sample(args, root=root, progress=lambda _: None)
                audit_path = next((root / 'artifacts/cache').glob('full_audit_*.json'))
                sample_path = root / 'artifacts/cache/sample_43_seed31.npz'
                metadata_path = sample_path.with_suffix('.json')
                cached_files = {
                    path: preparation.fingerprint(path)
                    for path in (audit_path, sample_path, metadata_path)
                }
                if not reuse_audit:
                    audit_path.rename(root / 'saved_audit.json')
                if not reuse_sample:
                    args.sample_size = 57
                with patch('criteo_uplift.preparation.scan_raw_data', wraps=preparation.scan_raw_data) as scan:
                    sample, audit, _, prepared = preparation.prepare_sample(args, root=root, progress=lambda _: None)
                self.assertEqual(scan.call_count, int(not (reuse_sample and reuse_audit)))
                if scan.called:
                    self.assertEqual(scan.call_args.kwargs['sample_enabled'], not reuse_sample)
                    self.assertEqual(scan.call_args.kwargs['audit_enabled'], not reuse_audit)
                    self.assertFalse(scan.call_args.kwargs['diagnostics_enabled'])
                current_sample = root / f'artifacts/cache/sample_{args.sample_size}_seed31.npz'
                self.assertEqual(json.loads(current_sample.with_suffix('.json').read_text()), prepared['sample_report'])
                self.assertEqual(json.loads(audit_path.read_text())['audit'], audit)
                if reuse_sample:
                    for path in (sample_path, metadata_path):
                        self.assertEqual(preparation.fingerprint(path), cached_files[path])
                if reuse_audit:
                    self.assertEqual(preparation.fingerprint(audit_path), cached_files[audit_path])
                self.assert_audit_close(audit, expected_audit)
                self.assertEqual(prepared['sample_report']['sample_n'], len(sample))
                self.assertNotIn('audit', prepared['sample_report'])
                for key in ('sample_n', 'sample_seed', 'sampling'):
                    self.assertNotIn(key, audit)

    def test_incomplete_sample_metadata_is_regenerated_by_scan(self):
        #-------------
        # 函数名：UnifiedScanTests.test_incomplete_sample_metadata_is_regenerated_by_scan
        # 功能：验证 Sample Metadata 字段不完整时会通过重新扫描重新生成，而不误用旧说明。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        original, original_audit, source, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        cache = self.root / 'artifacts/cache/sample_43_seed31.npz'
        metadata = cache.with_suffix('.json')
        saved_audit = json.loads(next((self.root / 'artifacts/cache').glob('full_audit_*.json')).read_text())['audit']
        metadata.write_text(json.dumps({
            'source': source, 'sample_size': args.sample_size, 'seed': args.seed,
            'audit': saved_audit,
        }))
        with patch('criteo_uplift.preparation.scan_raw_data', wraps=preparation.scan_raw_data) as scan:
            sample, audit, _, prepared = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertEqual(scan.call_args.kwargs['sample_size'], 43)
        self.assertFalse(scan.call_args.kwargs['audit_enabled'])
        pd.testing.assert_frame_equal(sample, original)
        self.assert_audit_close(audit, original_audit)
        self.assertEqual(json.loads(metadata.read_text()), prepared['sample_report'])
        self.assertNotIn('audit', prepared['sample_report'])

    def test_audit_cache_survives_size_seed_changes_and_invalidates_by_source_and_content(self):
        #-------------
        # 函数名：UnifiedScanTests.test_audit_cache_survives_size_seed_changes_and_invalidates_by_source_and_content
        # 功能：验证 Audit Cache 不因抽样规模或 seed 改变而失效，但会因数据来源变化或内容不完整而失效。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        _, original_audit, _, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        audit_path = next((self.root / 'artifacts/cache').glob('full_audit_*.json'))
        original_fingerprint = preparation.fingerprint(audit_path)
        for size, seed in ((61, 31), (61, 32), (0, 32), (5000, 33)):
            args.sample_size, args.seed = size, seed
            with patch('criteo_uplift.preparation.scan_raw_data', wraps=preparation.scan_raw_data) as scan:
                sample, audit, _, prepared = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
            self.assertFalse(scan.call_args.kwargs['audit_enabled'])
            self.assertEqual(scan.call_args.kwargs['sample_size'], size)
            self.assert_audit_close(audit, original_audit)
            self.assertEqual(preparation.fingerprint(audit_path), original_fingerprint)
            report = prepared['sample_report']
            self.assertEqual(report['seed'], seed)
            self.assertEqual(report['sample_size'], size)
            self.assertEqual(report['sample_n'], min(size, len(self.raw)) if size else len(self.raw))
            self.assertEqual(report['sample_n'], len(sample))
        valid = json.loads(audit_path.read_text())
        for invalid in (
            {**valid, 'audit': {}},
            {**valid, 'source': {'sha256': 'another_source'}},
        ):
            audit_path.write_text(json.dumps(invalid))
            with patch('criteo_uplift.preparation.scan_raw_data', wraps=preparation.scan_raw_data) as scan:
                _, audit, _, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
            self.assertTrue(scan.call_args.kwargs['audit_enabled'])
            self.assertFalse(scan.call_args.kwargs['sample_enabled'])
            self.assert_audit_close(audit, original_audit)

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

    def test_missing_duplicate_fields_rebuild_audit_without_resampling(self):
        #-------------
        # 函数名：UnifiedScanTests.test_missing_duplicate_fields_rebuild_audit_without_resampling
        # 功能：验证 Audit Cache 缺少重复行统计字段时会重建审计，但复用有效建模样本缓存。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        sample, original, _, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        path = next((self.root / 'artifacts/cache').glob('full_audit_*.json'))
        saved = json.loads(path.read_text())
        for field in ('duplicate_row_fraction', 'duplicate_fraction_denominator', 'duplicate_count_definition'):
            saved['audit'].pop(field)
        path.write_text(json.dumps(saved))
        cache = self.root / 'artifacts/cache/sample_43_seed31.npz'
        before = cache.read_bytes()
        with patch('criteo_uplift.preparation.scan_raw_data', wraps=preparation.scan_raw_data) as scan:
            reused, audit, _, result = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertTrue(scan.call_args.kwargs['audit_enabled'])
        self.assertFalse(scan.call_args.kwargs['sample_enabled'])
        self.assertEqual(cache.read_bytes(), before)
        pd.testing.assert_frame_equal(sample, reused)
        self.assert_audit_close(audit, original)
        self.assertEqual(result['sample_report']['group_summary'][-1]['group'], 'Total')
        self.assertEqual(json.loads(cache.with_suffix('.json').read_text())['group_summary'][-1]['group'], 'Total')

    def test_source_change_invalidates_both_caches(self):
        #-------------
        # 函数名：UnifiedScanTests.test_source_change_invalidates_both_caches
        # 功能：验证原始 CSV 内容指纹变化时 Sample Cache 和 Audit Cache 均失效。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        _, _, before_source, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        changed = pd.read_csv(self.path)
        changed.loc[0, 'f2'] += 1
        changed.to_csv(self.path, index=False)
        with patch('criteo_uplift.preparation.scan_raw_data', wraps=preparation.scan_raw_data) as scan:
            _, _, after_source, _ = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertNotEqual(before_source['sha256'], after_source['sha256'])
        self.assertEqual(scan.call_args.kwargs['sample_size'], 43)
        self.assertTrue(scan.call_args.kwargs['audit_enabled'])
        self.assertEqual(len(list((self.root / 'artifacts/cache').glob('full_audit_*.json'))), 2)


    def test_preparation_all_seven_task_combinations(self):
        #-------------
        # 七种开关组合各自冷启动一次，随后只读取已启用任务的缓存。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        before = preparation.fingerprint(self.path)
        for sample_on, audit_on, diagnostics_on in product((False, True), repeat=3):
            if not any((sample_on, audit_on, diagnostics_on)):
                continue
            with self.subTest(sample=sample_on, audit=audit_on, diagnostics=diagnostics_on):
                root = self.root / f'tasks_{sample_on}_{audit_on}_{diagnostics_on}'
                args = SimpleNamespace(data=self.path, seed=31)
                # 未启用任务不需要其专用参数，也不能读取其缓存。
                if sample_on:
                    args.sample_size = 43
                if diagnostics_on:
                    args.c2st_sample, args.permutations = 20, 1
                switches = dict(sample_enabled=sample_on, audit_enabled=audit_on,
                                diagnostics_enabled=diagnostics_on)
                with patch.object(raw_scan.pd, 'read_csv', wraps=pd.read_csv) as read:
                    first = preparation.prepare_sample(args, root=root, chunk_size=17,
                                                       progress=lambda _: None, **switches)
                self.assertEqual(sum('chunksize' in call.kwargs for call in read.call_args_list), 1)
                with patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('不应解析 CSV')):
                    second = preparation.prepare_sample(args, root=root, chunk_size=53,
                                                        progress=lambda _: None, **switches)
                for result, status in ((first, 'computed'), (second, 'cache')):
                    metadata = result[3]['metadata']
                    for name, enabled in switches.items():
                        self.assertEqual(metadata[name], enabled)
                    self.assertEqual(metadata['model_sample_enabled'], sample_on)
                    self.assertEqual(metadata['c2st_sample_enabled'], diagnostics_on)
                    self.assertEqual(metadata['task_status'], {
                        name: status if enabled else 'disabled'
                        for name, enabled in zip(('sample', 'audit', 'diagnostics'),
                                                 (sample_on, audit_on, diagnostics_on))
                    })
                    self.assertEqual(metadata['sample_n'], 20 if diagnostics_on else None)
                    self.assertEqual(result[0] is not None, sample_on)
                    self.assertEqual(result[1] is not None, audit_on)
                    self.assertEqual(result[3]['balance'] is not None, diagnostics_on)
                    self.assertEqual(result[3]['c2st_sample'] is not None, diagnostics_on)
                    self.assertEqual(result[3]['srm'] is not None, diagnostics_on)
                    self.assertNotIn('sample', result[3])
                if sample_on:
                    pd.testing.assert_frame_equal(first[0], second[0], check_exact=True)
                if audit_on:
                    self.assertEqual(first[1], second[1])
                if diagnostics_on:
                    pd.testing.assert_frame_equal(first[3]['balance'], second[3]['balance'], check_exact=True)
                    pd.testing.assert_frame_equal(first[3]['c2st_sample'], second[3]['c2st_sample'], check_exact=True)
                    self.assertEqual(first[3]['srm'], second[3]['srm'])
        self.assertEqual(preparation.fingerprint(self.path), before)

    def test_all_eight_independent_cache_hit_patterns(self):
        #-------------
        # 逐一失效三个缓存，验证命中任务不重算、不改写，未命中任务共用一次扫描。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        for sample_hit, audit_hit, diagnostics_hit in product((False, True), repeat=3):
            with self.subTest(sample=sample_hit, audit=audit_hit, diagnostics=diagnostics_hit):
                root = self.root / f'hits_{sample_hit}_{audit_hit}_{diagnostics_hit}'
                args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
                expected = preparation.prepare_sample(args, root=root, progress=lambda _: None)
                cache = root / 'artifacts/cache'
                manifests = [cache / 'sample_43_seed31.json', next(cache.glob('full_audit_*.json')),
                             next(cache.glob('full_diagnostics_*.json'))]
                hit_files = {}
                for hit, manifest in zip((sample_hit, audit_hit, diagnostics_hit), manifests):
                    if hit:
                        for file in (manifest, manifest.with_suffix('.npz')):
                            if file.exists():
                                hit_files[file] = preparation.fingerprint(file)
                    else:
                        manifest.unlink()
                with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan, \
                     patch.object(raw_scan.pd, 'read_csv', wraps=pd.read_csv) as read:
                    actual = preparation.prepare_sample(args, root=root, progress=lambda _: None)
                misses = not all((sample_hit, audit_hit, diagnostics_hit))
                self.assertEqual(scan.call_count, int(misses))
                self.assertEqual(sum('chunksize' in c.kwargs for c in read.call_args_list), int(misses))
                if misses:
                    self.assertEqual(scan.call_args.kwargs['sample_enabled'], not sample_hit)
                    self.assertEqual(scan.call_args.kwargs['audit_enabled'], not audit_hit)
                    self.assertEqual(scan.call_args.kwargs['diagnostics_enabled'], not diagnostics_hit)
                    self.assertEqual(scan.call_args.kwargs['per_arm'], None if diagnostics_hit else 10)
                else:
                    read.assert_not_called()
                metadata = actual[3]['metadata']
                for name in ('sample_enabled', 'audit_enabled', 'diagnostics_enabled'):
                    self.assertTrue(metadata[name])
                self.assertEqual(metadata['task_status'], {
                    name: 'cache' if hit else 'computed'
                    for name, hit in zip(('sample', 'audit', 'diagnostics'),
                                         (sample_hit, audit_hit, diagnostics_hit))
                })
                for file, before in hit_files.items():
                    self.assertEqual(preparation.fingerprint(file), before)
                pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
                self.assert_audit_close(actual[1], expected[1])
                pd.testing.assert_frame_equal(actual[3]['balance'], expected[3]['balance'], check_exact=True)
                pd.testing.assert_frame_equal(actual[3]['c2st_sample'], expected[3]['c2st_sample'], check_exact=True)
                self.assertEqual(actual[3]['srm'], expected[3]['srm'])

    def test_fixed_pre_refactor_reference_and_c2st_results(self):
        #-------------
        # 与重构前保存的统计和行号逐项对照，并验证实际 C2ST 检验输出一致。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
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
                self.assertEqual({k: v for k, v in scanned['srm'].items() if k != 'source_sha256'}, reference['srm'])
        # 基准行号来自旧实现；从原始 float64 表恢复相同输入，独立运行真实分类器。
        expected_sample = self.raw.iloc[reference['c2st_ids']]
        expected = diagnostics.c2st(expected_sample, FEATURES, sample_size=20, permutations=2, seed=31)
        actual = diagnostics.c2st(scanned['c2st_sample'], FEATURES, sample_size=20, permutations=2, seed=31)
        for key in ('summary', 'null_auc'):
            pd.testing.assert_frame_equal(actual[key], expected[key], check_exact=True)

    def test_diagnostics_cache_settings_and_payload_validation(self):
        #-------------
        # 诊断不依赖建模规模和置换次数，来源、抽样参数和缓存损坏会使其独立失效。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        args.sample_size, args.permutations = 57, 2
        with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan:
            preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertFalse(scan.call_args.kwargs['diagnostics_enabled'])
        manifest = next((self.root / 'artifacts/cache').glob('full_diagnostics_*.json'))
        valid = json.loads(manifest.read_text())
        self.assertNotIn('diagnostics_cache_version', valid)
        for invalid in ({**valid, 'seed': 99}, {**valid, 'sample_size': 40},
                        {**valid, 'source': {'sha256': '无效来源'}}, {**valid, 'metadata': {}}):
            manifest.write_text(json.dumps(invalid))
            with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan:
                preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
            self.assertFalse(scan.call_args.kwargs['sample_enabled'])
            self.assertFalse(scan.call_args.kwargs['audit_enabled'])
            self.assertTrue(scan.call_args.kwargs['diagnostics_enabled'])
        manifest.with_suffix('.npz').write_bytes(b'corrupt')
        with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan:
            preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertTrue(scan.call_args.kwargs['diagnostics_enabled'])
        args.c2st_sample = 40
        with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan:
            preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertEqual(scan.call_args.kwargs['per_arm'], 20)
        self.assertFalse(scan.call_args.kwargs['sample_enabled'])
        self.assertFalse(scan.call_args.kwargs['audit_enabled'])

    def test_diagnostics_cache_zlib_error_rebuilds_only_diagnostics(self):
        #-------------
        # 损坏压缩数据但保持载荷指纹匹配，验证真实 zlib 解压异常触发独立重建。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；验证一次扫描完成修复，且有效的 Sample、Audit 缓存不被改写。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        expected = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        directory = self.root / 'artifacts/cache'
        payload = next(directory.glob('full_diagnostics_*.npz'))
        manifest = payload.with_suffix('.json')
        preserved = {
            file: preparation.fingerprint(file)
            for file in directory.iterdir() if file not in (payload, manifest)
        }
        source = preparation.fingerprint(self.path)
        with ZipFile(payload) as archive:
            entry = archive.getinfo('X.npy')
            self.assertEqual(entry.compress_type, ZIP_DEFLATED)
            offset = entry.header_offset
        data = bytearray(payload.read_bytes())
        name_length = int.from_bytes(data[offset + 26:offset + 28], 'little')
        extra_length = int.from_bytes(data[offset + 28:offset + 30], 'little')
        compressed_start = offset + 30 + name_length + extra_length
        # DEFLATE 的 BTYPE=3 为保留类型，可稳定触发 zlib.error，而不是普通 ZIP 校验失败。
        data[compressed_start] = (data[compressed_start] & ~6) | 6
        payload.write_bytes(data)
        saved = json.loads(manifest.read_text())
        saved['payload_sha256'] = preparation.fingerprint(payload)['sha256']
        manifest.write_text(json.dumps(saved))
        with np.load(payload, allow_pickle=False) as arrays:
            with self.assertRaises(ZlibError):
                arrays['X']
        with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan, \
             patch.object(raw_scan.pd, 'read_csv', wraps=pd.read_csv) as read:
            actual = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertEqual(scan.call_count, 1)
        self.assertFalse(scan.call_args.kwargs['sample_enabled'])
        self.assertFalse(scan.call_args.kwargs['audit_enabled'])
        self.assertTrue(scan.call_args.kwargs['diagnostics_enabled'])
        self.assertEqual(sum('chunksize' in call.kwargs for call in read.call_args_list), 1)
        pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
        self.assertEqual(actual[1], expected[1])
        for key in ('balance', 'c2st_sample'):
            pd.testing.assert_frame_equal(actual[3][key], expected[3][key], check_exact=True)
        self.assertEqual(actual[3]['srm'], expected[3]['srm'])
        for file, before in preserved.items():
            self.assertEqual(preparation.fingerprint(file), before)
        self.assertEqual(preparation.fingerprint(self.path), source)
        with patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('修复后应全部命中')):
            preparation.prepare_sample(args, root=self.root, progress=lambda _: None)

    def test_diagnostics_cache_preserves_infinite_smd(self):
        #-------------
        # 分组内零方差、组间均值不同的无限 SMD 不应被 JSON 缓存变成空值。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        frame = self.raw.copy()
        frame['f1'] = frame.treatment
        frame.to_csv(self.path, index=False)
        args = SimpleNamespace(data=self.path, seed=31, c2st_sample=20, permutations=1)
        options = dict(root=self.root, sample_enabled=False, audit_enabled=False, progress=lambda _: None)
        first = preparation.prepare_sample(args, **options)[3]
        with patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('不应解析 CSV')):
            cached = preparation.prepare_sample(args, **options)[3]
        self.assertTrue(np.isinf(cached['balance'].set_index('feature').loc['f1', 'smd']))
        pd.testing.assert_frame_equal(first['balance'], cached['balance'], check_exact=True)

    def test_scan_resources_close_on_invalid_late_chunk(self):
        #-------------
        # 末尾非法行仍会失败，并关闭读取器和全部诊断临时文件，不改动来源。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        from tempfile import TemporaryFile
        files, readers = [], []
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
                readers.append(result)
            return result
        frame = self.raw.copy()
        frame.loc[len(frame) - 1, 'conversion'] = 2
        frame.to_csv(self.path, index=False)
        before = preparation.fingerprint(self.path)
        with patch.object(raw_scan, 'TemporaryFile', side_effect=temporary_file), \
             patch.object(raw_scan.pd, 'read_csv', side_effect=read_csv):
            with self.assertRaises(ValueError):
                self.scan()
        self.assertEqual(len(files), len(FEATURES) + 1)
        self.assertTrue(all(file.closed for file in files))
        self.assertTrue(all(reader.handles.handle.closed for reader in readers))
        self.assertEqual(preparation.fingerprint(self.path), before)

    def test_disabled_and_invalid_switches_fail_before_csv_parsing(self):
        #-------------
        # 非法开关、全关闭以及非法分块参数均在读取 CSV 前报错。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        with patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('不应解析 CSV')):
            for options in ({'sample_enabled': 1}, {'audit_enabled': '是'}, {'diagnostics_enabled': 0},
                            {'sample_enabled': False, 'audit_enabled': False, 'diagnostics_enabled': False},
                            {'chunk_size': 0}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    preparation.prepare_sample(args, root=self.root, **options)


    def test_sample_npz_corruption_recovers_in_one_shared_scan(self):
        #-------------
        # 保留正确说明，覆盖文件损坏、缺字段、形状错误及非法数据；与其他缓存失效共享一次扫描。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        expected = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        directory = self.root / 'artifacts/cache'
        path = directory / 'sample_43_seed31.npz'
        manifest = path.with_suffix('.json')
        metadata = manifest.read_bytes()
        valid_bytes = path.read_bytes()
        with np.load(path, allow_pickle=False) as arrays:
            valid = {name: arrays[name] for name in arrays.files}
        audit_path = next(directory.glob('full_audit_*.json'))
        diagnostic_path = next(directory.glob('full_diagnostics_*.json'))
        source = preparation.fingerprint(self.path)
        cases = ('broken', 'truncated', 'missing', 'feature_shape', 'label_shape',
                 'nonfinite', 'binary', 'duplicate_ids', 'out_of_bounds', 'wrong_rows')
        for case, audit_hit, diagnostics_hit in product(cases, (False, True), (False, True)):
            with self.subTest(case=case, audit_hit=audit_hit, diagnostics_hit=diagnostics_hit):
                arrays = {name: values.copy() for name, values in valid.items()}
                if case == 'broken':
                    path.write_bytes(b'broken npz')
                elif case == 'truncated':
                    path.write_bytes(valid_bytes[:len(valid_bytes) // 2])
                else:
                    if case == 'missing':
                        arrays.pop('row_id')
                    elif case == 'feature_shape':
                        arrays['X'] = arrays['X'][:, :-1]
                    elif case == 'label_shape':
                        arrays['labels'] = arrays['labels'][:-1]
                    elif case == 'nonfinite':
                        arrays['X'][0, 0] = np.nan
                    elif case == 'binary':
                        arrays['labels'][0, 0] = 2
                    elif case == 'duplicate_ids':
                        arrays['row_id'][0] = arrays['row_id'][1]
                    elif case == 'out_of_bounds':
                        arrays['row_id'][-1] = len(self.raw)
                    elif case == 'wrong_rows':
                        arrays = {name: values[:-1] for name, values in arrays.items()}
                    np.savez_compressed(path, **arrays)
                self.assertEqual(manifest.read_bytes(), metadata)
                preserved = {}
                for hit, cache in ((audit_hit, audit_path), (diagnostics_hit, diagnostic_path)):
                    if hit:
                        for file in (cache, cache.with_suffix('.npz')):
                            if file.exists():
                                preserved[file] = preparation.fingerprint(file)
                    else:
                        cache.unlink()
                with patch.object(preparation, 'scan_raw_data', wraps=raw_scan.scan_raw_data) as scan, \
                     patch.object(raw_scan.pd, 'read_csv', wraps=pd.read_csv) as read, \
                     patch.object(preparation, 'write_result', wraps=preparation.write_result) as write:
                    actual = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
                self.assertEqual(scan.call_count, 1)
                self.assertTrue(scan.call_args.kwargs['sample_enabled'])
                self.assertEqual(scan.call_args.kwargs['audit_enabled'], not audit_hit)
                self.assertEqual(scan.call_args.kwargs['diagnostics_enabled'], not diagnostics_hit)
                self.assertEqual(sum('chunksize' in c.kwargs for c in read.call_args_list), 1)
                self.assertTrue(any(Path(c.args[1]).resolve() == manifest.resolve() for c in write.call_args_list))
                pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=True)
                pd.testing.assert_frame_equal(load_sample(path), expected[0], check_exact=True)
                self.assertEqual(json.loads(manifest.read_text()), actual[3]['sample_report'])
                self.assert_audit_close(actual[1], expected[1])
                self.assertEqual(actual[3]['metadata']['task_status']['sample'], 'computed')
                for file, before in preserved.items():
                    self.assertEqual(preparation.fingerprint(file), before)
                with patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('重建后应全部命中')):
                    preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        self.assertEqual(preparation.fingerprint(self.path), source)

    def test_metadata_uses_current_switches_when_reusing_previous_results(self):
        #-------------
        # 先生成全任务缓存，再切换七种组合，返回说明必须反映本次开关，且不改写缓存。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        preparation.prepare_sample(args, root=self.root, progress=lambda _: None)
        files = {p: preparation.fingerprint(p) for p in (self.root / 'artifacts/cache').iterdir()}
        for enabled in product((False, True), repeat=3):
            if not any(enabled):
                continue
            switches = dict(zip(('sample_enabled', 'audit_enabled', 'diagnostics_enabled'), enabled))
            with self.subTest(**switches), patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('不应扫描')):
                for name, value in switches.items():
                    setattr(args, name, value)
                prepared = preparation.prepare_sample(args, root=self.root, progress=lambda _: None)[3]
            metadata = prepared['metadata']
            for name, value in switches.items():
                self.assertEqual(metadata[name], value)
            self.assertEqual(metadata['task_status'], {
                name: 'cache' if value else 'disabled'
                for name, value in zip(('sample', 'audit', 'diagnostics'), enabled)
            })
            self.assertEqual(metadata['c2st_sample_enabled'], enabled[2])
            self.assertEqual(metadata['sample_n'], 20 if enabled[2] else None)
        for file, before in files.items():
            self.assertEqual(preparation.fingerprint(file), before)

    def test_scan_requires_explicit_task_switches(self):
        #-------------
        # None 不能隐式关闭建模或 C2ST；关闭 Diagnostics 后其参数不能单独触发任何诊断计算。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        with patch.object(raw_scan.pd, 'read_csv', side_effect=AssertionError('参数错误不得读取来源')):
            for options in ({'sample_size': None}, {'per_arm': None}, {'sample_enabled': 1},
                            {'audit_enabled': '是'}, {'diagnostics_enabled': 0},
                            {'sample_enabled': False, 'audit_enabled': False, 'diagnostics_enabled': False}):
                with self.subTest(**options), self.assertRaises(ValueError):
                    self.scan(**options)
        with patch.object(raw_scan, 'DiagnosticsAccumulator', side_effect=AssertionError('诊断已关闭')):
            result = self.scan(sample_enabled=False, sample_size=None,
                               diagnostics_enabled=False, per_arm=1)
        self.assertIsNone(result['sample'])
        self.assertIsNone(result['c2st_sample'])
        self.assertIsNone(result['balance'])
        self.assertIsNone(result['srm'])
        self.assertEqual(result['audit']['n_rows'], len(self.raw))

    def test_audit_reference_tolerance_rejects_real_regressions(self):
        #-------------
        # 原有容差可处理浮点舍入，但必须拒绝超出容差的统计变化及精确计数字段变化。
        # 输入参数：
        #     - self：当前测试实例，提供临时原始数据和缓存目录。
        # 输出参数：
        #     - 无返回值；通过断言验证预期行为，失败时抛出 AssertionError。
        #-------------
        reference = json.loads((Path(__file__).parent / 'fixtures/raw_scan_reference.json').read_text())['audit']
        for chunk_size in (1, 17, 53, 160):
            self.assert_audit_close(self.scan(chunk_size=chunk_size, diagnostics_enabled=False)['audit'], reference)
        for field in ('feature_stats', 'n_rows'):
            changed = json.loads(json.dumps(reference))
            if field == 'feature_stats':
                changed[field][0]['smd'] += 1e-3
            else:
                changed[field] += 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.assert_audit_close(changed, reference)


if __name__ == '__main__':
    unittest.main()
