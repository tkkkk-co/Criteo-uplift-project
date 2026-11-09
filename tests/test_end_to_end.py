# 项目测试：验证完整实验流程。

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest
import numpy as np
import pandas as pd
from criteo_uplift.data import FEATURES, fingerprint
from run_pipeline import run, update_diagnostics_only


#-----------------------------------------------------------------------------------------------------------------------
# 类：EndToEndTests —— 验证完整实验流程。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证完整实验流程。
#-----------------------------------------------------------------------------------------------------------------------
class EndToEndTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_complete_pipeline_outputs_and_frozen_selection —— 运行合成数据完整流程并检查输出及验证集冻结选择。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：运行合成数据完整流程并检查输出及验证集冻结选择。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_complete_pipeline_outputs_and_frozen_selection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(814)
            x = rng.normal(size=(4000, 12))
            t = rng.binomial(1, 0.8, len(x))
            probability = 1 / (
                1 + np.exp(2.4 - 0.8 * x[:, 0] - 0.7 * t * (x[:, 1] > 0))
            )
            y = rng.binomial(1, probability)
            frame = pd.DataFrame(x, columns=FEATURES)
            frame['treatment'] = t
            frame['conversion'] = y
            frame['visit'] = np.maximum(y, rng.binomial(1, 0.2, len(x)))
            frame['exposure'] = t * rng.binomial(1, 0.5, len(x))
            path = root / 'synthetic.csv'
            frame.to_csv(path, index=False)
            before = fingerprint(path)
            args = SimpleNamespace(
                data=path,
                sample_size=4000,
                seed=814,
                outcome='conversion',
                primary_budget=0.2,
                permutations=2,
                c2st_sample=1000,
                bootstrap=5,
                trees=5,
                output=root / 'run',
            )
            with patch('run_pipeline.ROOT', root), patch(
                'criteo_uplift.diagnostics.pd.read_csv', wraps=pd.read_csv,
            ) as read_csv:
                out = run(args)
            raw_scans = [
                call for call in read_csv.call_args_list
                if call.args and Path(call.args[0]) == path and 'chunksize' in call.kwargs
            ]
            self.assertEqual(len(raw_scans), 1)
            for name in (
                'data_preparation/audit.json', 'data_preparation/sample_report.json',
                'data_preparation/split_summary.csv', 'data_preparation/test_row_ids.csv',
                'diagnostics/learnability.csv', 'diagnostics/balance.csv',
                'models/selection.json', 'models/t_xgboost.joblib',
                'evaluation/test_policies.csv', 'evaluation/predictions.npz',
                'figures/sample_features.png', 'figures/c2st_null.png',
            ):
                self.assertTrue((out / name).is_file(), name)
            summary = pd.read_csv(out / 'data_preparation/split_summary.csv')
            self.assertEqual(len(summary), 6)
            self.assertEqual(summary.n.sum(), len(frame))
            source_by_id = frame.copy()
            source_by_id.index.name = 'row_id'
            all_ids = []
            for name in ('train', 'validation', 'test'):
                ids = pd.read_csv(out / f'data_preparation/{name}_row_ids.csv').row_id
                all_ids.extend(ids.tolist())
                members = source_by_id.loc[ids]
                for arm in (0, 1):
                    rows = members[members.treatment == arm]
                    actual = summary[(summary.split == name) & (summary.treatment == arm)].iloc[0]
                    self.assertEqual(actual.n, len(rows))
                    self.assertEqual(actual.conversion_positive, int(rows.conversion.sum()))
                    self.assertEqual(actual.visit_positive, int(rows.visit.sum()))
            self.assertEqual(sorted(all_ids), list(range(len(frame))))
            from criteo_uplift.evaluation import ate_summary
            ate_full = pd.read_csv(out / 'diagnostics/ate_full.csv')
            for outcome in ('conversion', 'visit'):
                expected = ate_summary(frame[outcome], frame.treatment)
                actual = ate_full[ate_full.outcome == outcome].iloc[0]
                for key, value in expected.items():
                    if key in actual.index and isinstance(value, (int, float)):
                        self.assertAlmostEqual(actual[key], value)
            self.assertFalse((out / 'splits.csv').exists())
            self.assertFalse((out / 'split_summary.csv').exists())
            manifest = json.loads(((out / 'manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'complete')
            audit = json.loads(((out / 'data_preparation' / 'audit.json')).read_text())
            sample_report = json.loads(((out / 'data_preparation' / 'sample_report.json')).read_text())
            self.assertNotIn('audit', sample_report)
            self.assertEqual(sample_report['sample_n'], 4000)
            self.assertEqual(sample_report['seed'], args.seed)
            self.assertEqual(sample_report['source_n'], audit['n_rows'])
            for key in ('sample_n', 'sample_seed', 'sampling'):
                self.assertNotIn(key, audit)

            self.assertTrue(manifest['selection']['frozen_before_test'])
            val = pd.read_csv((out / 'models' / 'validation_policies.csv'))
            winner = (
                val.sort_values('incremental_per_100k', ascending=False, kind='stable')
                .iloc[0]
                .policy
            )
            self.assertEqual(manifest['selection']['selected_policy'], winner)
            ranking = pd.read_csv((out / 'evaluation' / 'test_ranking.csv'))
            self.assertIn('qini_score_traditional', ranking)
            policies = pd.read_csv((out / 'evaluation' / 'test_policies.csv'))
            self.assertEqual(
                policies[policies.budget == 1].incremental_per_100k.nunique(), 1
            )
            self.assertTrue((out / 'report.html').stat().st_size > 1000)
            self.assertEqual(len(list((out / 'figures').glob('*.png'))), 7)
            balance = pd.read_csv((out / 'diagnostics' / 'balance.csv'))
            self.assertTrue(balance['scope'].eq('full_published_dataset').all())
            self.assertTrue((balance.n_treatment + balance.n_control == len(frame)).all())
            diagnostic = json.loads(((out / 'diagnostics' / 'c2st_details.json')).read_text())
            self.assertEqual(diagnostic['scope'], 'full_source_arm_balanced_sample')
            self.assertEqual(diagnostic['n_treatment'], 500)
            self.assertEqual(diagnostic['n_control'], 500)
            self.assertEqual(fingerprint(path), before)


    def test_diagnostics_refresh_preserves_frozen_results_and_saved_ate(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            root = Path(folder)
            rng = np.random.default_rng(28)
            frame = pd.DataFrame(rng.normal(size=(1000, 12)), columns=FEATURES)
            frame['treatment'] = rng.binomial(1, .7, len(frame))
            frame['conversion'] = rng.binomial(1, .2, len(frame))
            frame['visit'] = np.maximum(frame.conversion, rng.binomial(1, .3, len(frame)))
            frame['exposure'] = frame.treatment
            path = root / 'synthetic.csv'
            frame.to_csv(path, index=False)
            args = SimpleNamespace(data=path, sample_size=800, seed=28, outcome='conversion',
                primary_budget=.2, permutations=1, c2st_sample=100, bootstrap=2, trees=2, output=root/'run')
            with patch('run_pipeline.ROOT', root):
                out = run(args)
            protected = {p: p.read_bytes() for p in out.rglob('*') if p.is_file() and (
                p.parent.name in ('models', 'evaluation') or p.name.endswith('_row_ids.csv') or p.name=='split_summary.csv')}
            ate_path = out/'diagnostics/ate_full.csv'
            before = pd.read_csv(ate_path, dtype={'p_value_text':'string'}, float_precision='round_trip')
            cache = next((root/'artifacts/cache').glob('full_audit_*.json'))
            old = json.loads(cache.read_text())
            old['audit'].pop('duplicate_row_fraction')
            cache.write_text(json.dumps(old))
            update_diagnostics_only(args, root=root, progress=lambda _: None)
            after = pd.read_csv(ate_path, dtype={'p_value_text':'string'}, float_precision='round_trip')
            pd.testing.assert_frame_equal(before, after, check_exact=True)
            for p, content in protected.items():
                self.assertEqual(p.read_bytes(), content, str(p))
            audit = json.loads((out/'data_preparation/audit.json').read_text())
            self.assertIn('duplicate_row_fraction', audit)
            self.assertEqual(audit['group_summary'][-1]['group'], 'Total')
            self.assertEqual(set(pd.read_csv(out/'diagnostics/c2st.csv').model), {'logistic','xgboost'})


if __name__ == '__main__':
    unittest.main()
