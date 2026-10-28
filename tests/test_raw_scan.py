from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from itertools import product
import json
import unittest

import numpy as np
import pandas as pd

from criteo_uplift import data, diagnostics
from criteo_uplift.diagnostics import covariate_balance, update_experiment_diagnostics
from criteo_uplift.pipeline_io import load_sample
from run_pipeline import prepare_sample


#-----------------------------------------------------------------------------------------------------------------------
# 类：UnifiedScanTests —— 验证统一扫描的统计与抽样。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证统一扫描的统计与抽样。
#-----------------------------------------------------------------------------------------------------------------------
class UnifiedScanTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：setUp —— 准备当前测试使用的数据和临时环境。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；测试数据保存在当前测试对象中。
    # 说明：准备当前测试使用的数据和临时环境。
    #-----------------------------------------------------------------------------------------------------------------------
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / 'raw.csv'
        rng = np.random.default_rng(9)
        frame = pd.DataFrame(rng.normal(size=(160, 12)), columns=data.FEATURES)
        for label in data.LABELS:
            frame[label] = rng.integers(0, 2, len(frame))
        frame['treatment'] = np.r_[np.zeros(40), np.ones(120)]
        frame['f0'] = 1 + np.arange(len(frame)) * 1e-10
        frame['f1'] = 7.0
        frame.iloc[-1] = frame.iloc[-2]  # 重复行仍参与审计、统计和抽样。
        frame.to_csv(self.path, index=False)
        self.raw = pd.read_csv(self.path, dtype=np.float64)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：scan —— 按测试配置调用统一扫描。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    #     **kwargs：传递给被调用函数的额外命名参数。
    # 输出：
    #     scanned：本次扫描的样本、审计和诊断结果。
    # 说明：按测试配置调用统一扫描。
    #-----------------------------------------------------------------------------------------------------------------------
    def scan(self, **kwargs):
        options = dict(sample_size=43, per_arm=10, seed=31, chunk_size=17, progress=lambda _: None)
        options.update(kwargs)
        return diagnostics.scan_raw_data(self.path, **options)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_one_csv_pass_preserves_both_sampling_streams_and_full_statistics —— 确认一次 CSV 遍历保留两套抽样结果和全量统计。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认一次 CSV 遍历保留两套抽样结果和全量统计。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_one_csv_pass_preserves_both_sampling_streams_and_full_statistics(self):
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
        np.testing.assert_array_equal(sample[data.FEATURES], self.raw.iloc[sample.row_id][data.FEATURES])
        self.assertEqual(sample.f0.dtype, np.dtype('float64'))
        self.assertEqual(scanned['sample'].f0.dtype, np.dtype('float32'))

        expected = covariate_balance(self.raw, data.FEATURES)
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_audit_only_variance_matches_full_diagnostics_for_near_constant_features —— 核对近常量特征在不同扫描模式下的方差和 SMD。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对近常量特征在不同扫描模式下的方差和 SMD。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_audit_only_variance_matches_full_diagnostics_for_near_constant_features(self):
        expected = self.scan()['audit']['feature_stats']
        for chunk_size in (13, 31, len(self.raw)):
            actual = self.scan(diagnostics_enabled=False, chunk_size=chunk_size)['audit']['feature_stats']
            for observed, reference in zip(actual, expected):
                with self.subTest(chunk_size=chunk_size, feature=reference['feature']):
                    for field in ('mean_0', 'mean_1', 'std_0', 'std_1', 'smd'):
                        np.testing.assert_allclose(observed[field], reference[field], rtol=2e-6, atol=1e-15)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_diagnostic_sample_independent_of_model_size_and_chunk_size —— 确认诊断抽样不受建模样本规模和分块大小影响。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认诊断抽样不受建模样本规模和分块大小影响。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_diagnostic_sample_independent_of_model_size_and_chunk_size(self):
        baseline = self.scan()
        for size in (None, 0, 200):
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
                    self.assertEqual(len(scanned['sample']), len(self.raw))
                    self.assertEqual(scanned['sample_report']['sample_size'], size)
                    self.assertEqual(scanned['sample_report']['sample_n'], len(self.raw))
                    self.assertEqual(
                        scanned['sample_report']['sampling'],
                        'all rows' if size == 0 else 'uniform without replacement, random-priority reservoir',
                    )

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_audit_diagnostics_and_two_samples_are_independent —— 确认审计、全量诊断和两套抽样可以独立启用。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认审计、全量诊断和两套抽样可以独立启用。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_audit_diagnostics_and_two_samples_are_independent(self):
        expected_balance = covariate_balance(self.raw, data.FEATURES)
        baseline = self.scan()
        for audit, checks, model_size, per_arm in product((False, True), (False, True), (None, 43), (None, 10)):
            options = dict(
                audit_enabled=audit, diagnostics_enabled=checks,
                sample_size=model_size, per_arm=per_arm,
            )
            with self.subTest(**options):
                if not any((audit, checks, model_size is not None, per_arm is not None)):
                    with self.assertRaisesRegex(ValueError, 'at least one'):
                        self.scan(**options)
                    continue
                scanned = self.scan(**options)
                if audit:
                    self.assertEqual(scanned['audit']['n_rows'], len(self.raw))
                    self.assertEqual(scanned['audit']['duplicate_row_fingerprints'], 1)
                    self.assertEqual(scanned['audit']['n_control'], 40)
                    self.assertEqual(len(scanned['audit']['feature_stats']), 12)
                    for group, (name, rows) in zip(scanned['audit']['group_summary'], (
                        ('Control', self.raw[self.raw.treatment == 0]),
                        ('Treatment', self.raw[self.raw.treatment == 1]),
                        ('总体', self.raw),
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
                if model_size is None:
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
                        ('总体', scanned['sample']),
                    )):
                        self.assertEqual(group['group'], name)
                        self.assertEqual(group['n'], len(rows))
                        self.assertEqual(group['fraction'], len(rows) / len(scanned['sample']))
                        self.assertEqual(group['conversion_positive'], int(rows.conversion.sum()))
                        self.assertAlmostEqual(group['conversion_rate'], float(rows.conversion.mean()))
                    self.assertNotIn('audit', report)
                if per_arm is None:
                    self.assertIsNone(scanned['c2st_sample'])
                    self.assertIsNone(scanned['metadata']['sample_n'])
                else:
                    pd.testing.assert_frame_equal(scanned['c2st_sample'], baseline['c2st_sample'])

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_source_changes_and_insufficient_arms_fail —— 确认扫描拒绝变化的输入来源和不足的组内人数。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认扫描拒绝变化的输入来源和不足的组内人数。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_source_changes_and_insufficient_arms_fail(self):
        source = data.fingerprint(self.path)
        with patch.object(diagnostics, 'fingerprint', side_effect=[source, {**source, 'sha256': 'changed'}]):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                self.scan()
        with self.assertRaisesRegex(ValueError, 'Need 41 rows per arm'):
            self.scan(per_arm=41)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_update_reuses_scan_and_rejects_mismatched_settings —— 确认诊断更新复用扫描并拒绝不一致的配置。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认诊断更新复用扫描并拒绝不一致的配置。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_update_reuses_scan_and_rejects_mismatched_settings(self):
        scanned = self.scan()
        out = self.root / 'diagnostics'
        out.mkdir()
        with patch('criteo_uplift.diagnostics.scan_raw_data', side_effect=AssertionError('second scan')):
            result = update_experiment_diagnostics(
                self.path, out, audit=scanned['audit'], source=scanned['source'],
                sample_size=20, permutations=1, seed=31, scanned=scanned, progress=lambda _: None,
            )
            self.assertEqual(result['full_n'], len(self.raw))
            for seed, size in ((32, 20), (31, 40)):
                with self.assertRaisesRegex(ValueError, '规模或种子'):
                    update_experiment_diagnostics(
                        self.path, out, audit=scanned['audit'], source=scanned['source'],
                        sample_size=size, permutations=1, seed=seed, scanned=scanned,
                    )
        self.assertEqual(json.loads((out / 'srm.json').read_text())['observed_ratio'], .75)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_prepare_sample_fresh_and_cached_paths_each_parse_csv_once —— 确认首次准备和缓存复用都只分块解析一次 CSV。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认首次准备和缓存复用都只分块解析一次 CSV。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_prepare_sample_fresh_and_cached_paths_each_parse_csv_once(self):
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        with patch('run_pipeline.ROOT', self.root), patch('run_pipeline.log'):
            with patch.object(diagnostics.pd, 'read_csv', wraps=pd.read_csv) as read:
                sample, audit, source, prepared = prepare_sample(args)
            self.assertEqual(sum('chunksize' in c.kwargs for c in read.call_args_list), 1)
            with patch.object(diagnostics.pd, 'read_csv', wraps=pd.read_csv) as read, patch.object(
                diagnostics.pd.util, 'hash_pandas_object', side_effect=AssertionError('repeated audit'),
            ):
                cached, cached_audit, cached_source, cached_diagnostics = prepare_sample(args)
            self.assertEqual(sum('chunksize' in c.kwargs for c in read.call_args_list), 1)
        pd.testing.assert_frame_equal(sample, cached)
        pd.testing.assert_frame_equal(prepared['c2st_sample'], cached_diagnostics['c2st_sample'])
        self.assertEqual(audit['n_rows'], cached_audit['n_rows'])
        self.assertEqual(source, cached_source)
        self.assertNotIn('sample', prepared)  # 不让后续诊断持有百万行建模表。

    def test_prepare_sample_rebuilds_without_metadata_even_with_legacy_files(self):
        args = SimpleNamespace(
            data=self.path, sample_size=1_000_000, seed=42,
            c2st_sample=20, permutations=1,
        )
        with patch('run_pipeline.ROOT', self.root), patch('run_pipeline.log'):
            sample, audit, source, _ = prepare_sample(args)
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
            with patch('criteo_uplift.preparation.load_sample', side_effect=AssertionError('old cache reused')):
                rebuilt, rebuilt_audit, rebuilt_source, _ = prepare_sample(args)

        pd.testing.assert_frame_equal(rebuilt, sample)
        pd.testing.assert_frame_equal(load_sample(cache), sample)
        self.assertEqual(rebuilt_audit, audit)
        self.assertEqual(rebuilt_source, source)
        self.assertEqual(json.loads(metadata.read_text()), info)
        self.assertEqual([p.read_bytes() for p in (legacy_source, legacy_audit)], legacy_bytes)


    def test_sample_and_audit_caches_are_reused_independently(self):
        for reuse_sample, reuse_audit in product((False, True), repeat=2):
            with self.subTest(reuse_sample=reuse_sample, reuse_audit=reuse_audit):
                root = self.root / f'combination_{reuse_sample}_{reuse_audit}'
                root.mkdir()
                args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
                with patch('run_pipeline.ROOT', root), patch('run_pipeline.log'):
                    _, expected_audit, _, _ = prepare_sample(args)
                    audit_path = next((root / 'artifacts/cache').glob('full_audit_*.json'))
                    sample_path = root / 'artifacts/cache/sample_43_seed31.npz'
                    metadata_path = sample_path.with_suffix('.json')
                    cached_files = {
                        path: data.fingerprint(path)
                        for path in (audit_path, sample_path, metadata_path)
                    }
                    if not reuse_audit:
                        audit_path.rename(root / 'saved_audit.json')
                    if not reuse_sample:
                        args.sample_size = 57
                    with patch('criteo_uplift.preparation.scan_raw_data', wraps=diagnostics.scan_raw_data) as scan:
                        sample, audit, _, prepared = prepare_sample(args)
                    self.assertEqual(scan.call_count, 1)
                    self.assertEqual(scan.call_args.kwargs['sample_size'], None if reuse_sample else 57)
                    self.assertEqual(scan.call_args.kwargs['audit_enabled'], not reuse_audit)
                    current_sample = root / f'artifacts/cache/sample_{args.sample_size}_seed31.npz'
                    self.assertEqual(json.loads(current_sample.with_suffix('.json').read_text()), prepared['sample_report'])
                    self.assertEqual(json.loads(audit_path.read_text())['audit'], audit)
                    if reuse_sample:
                        for path in (sample_path, metadata_path):
                            self.assertEqual(data.fingerprint(path), cached_files[path])
                    if reuse_audit:
                        self.assertEqual(data.fingerprint(audit_path), cached_files[audit_path])
                self.assertEqual(audit, expected_audit)
                self.assertEqual(prepared['sample_report']['sample_n'], len(sample))
                self.assertNotIn('audit', prepared['sample_report'])
                for key in ('sample_n', 'sample_seed', 'sampling'):
                    self.assertNotIn(key, audit)

    def test_incomplete_sample_metadata_is_regenerated_by_scan(self):
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        with patch('run_pipeline.ROOT', self.root), patch('run_pipeline.log'):
            original, original_audit, source, _ = prepare_sample(args)
            cache = self.root / 'artifacts/cache/sample_43_seed31.npz'
            metadata = cache.with_suffix('.json')
            saved_audit = json.loads(next((self.root / 'artifacts/cache').glob('full_audit_*.json')).read_text())['audit']
            metadata.write_text(json.dumps({
                'source': source, 'sample_size': args.sample_size, 'seed': args.seed,
                'audit': saved_audit,
            }))
            with patch('criteo_uplift.preparation.scan_raw_data', wraps=diagnostics.scan_raw_data) as scan:
                sample, audit, _, prepared = prepare_sample(args)
        self.assertEqual(scan.call_args.kwargs['sample_size'], 43)
        self.assertFalse(scan.call_args.kwargs['audit_enabled'])
        pd.testing.assert_frame_equal(sample, original)
        self.assertEqual(audit, original_audit)
        self.assertEqual(json.loads(metadata.read_text()), prepared['sample_report'])
        self.assertNotIn('audit', prepared['sample_report'])

    def test_audit_cache_survives_size_seed_changes_and_invalidates_by_source_and_version(self):
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        with patch('run_pipeline.ROOT', self.root), patch('run_pipeline.log'):
            _, original_audit, _, _ = prepare_sample(args)
            audit_path = next((self.root / 'artifacts/cache').glob('full_audit_*.json'))
            original_fingerprint = data.fingerprint(audit_path)
            for size, seed in ((61, 31), (61, 32), (0, 32), (5000, 33)):
                args.sample_size, args.seed = size, seed
                with patch('criteo_uplift.preparation.scan_raw_data', wraps=diagnostics.scan_raw_data) as scan:
                    sample, audit, _, prepared = prepare_sample(args)
                self.assertFalse(scan.call_args.kwargs['audit_enabled'])
                self.assertEqual(scan.call_args.kwargs['sample_size'], size)
                self.assertEqual(audit, original_audit)
                self.assertEqual(data.fingerprint(audit_path), original_fingerprint)
                report = prepared['sample_report']
                self.assertEqual(report['seed'], seed)
                self.assertEqual(report['sample_size'], size)
                self.assertEqual(report['sample_n'], min(size, len(self.raw)) if size else len(self.raw))
                self.assertEqual(report['sample_n'], len(sample))
            valid = json.loads(audit_path.read_text())
            for invalid in (
                {**valid, 'audit_cache_version': -1},
                {**valid, 'source': {'sha256': 'another_source'}},
            ):
                audit_path.write_text(json.dumps(invalid))
                with patch('criteo_uplift.preparation.scan_raw_data', wraps=diagnostics.scan_raw_data) as scan:
                    _, audit, _, _ = prepare_sample(args)
                self.assertTrue(scan.call_args.kwargs['audit_enabled'])
                self.assertIsNone(scan.call_args.kwargs['sample_size'])
                self.assertEqual(audit, original_audit)

    def test_source_change_invalidates_both_caches(self):
        args = SimpleNamespace(data=self.path, sample_size=43, seed=31, c2st_sample=20, permutations=1)
        with patch('run_pipeline.ROOT', self.root), patch('run_pipeline.log'):
            _, _, before_source, _ = prepare_sample(args)
            changed = pd.read_csv(self.path)
            changed.loc[0, 'f2'] += 1
            changed.to_csv(self.path, index=False)
            with patch('criteo_uplift.preparation.scan_raw_data', wraps=diagnostics.scan_raw_data) as scan:
                _, _, after_source, _ = prepare_sample(args)
        self.assertNotEqual(before_source['sha256'], after_source['sha256'])
        self.assertEqual(scan.call_args.kwargs['sample_size'], 43)
        self.assertTrue(scan.call_args.kwargs['audit_enabled'])
        self.assertEqual(len(list((self.root / 'artifacts/cache').glob('full_audit_*.json'))), 2)


if __name__ == '__main__':
    unittest.main()
