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
from run_pipeline import run


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
            ):
                self.assertTrue((out / name).is_file(), name)
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
            self.assertEqual(len(list((out / 'figures').glob('*.png'))), 5)
            balance = pd.read_csv((out / 'diagnostics' / 'balance.csv'))
            self.assertTrue(balance['scope'].eq('full_published_dataset').all())
            self.assertTrue((balance.n_treatment + balance.n_control == len(frame)).all())
            diagnostic = json.loads(((out / 'diagnostics' / 'c2st_details.json')).read_text())
            self.assertEqual(diagnostic['scope'], 'full_source_arm_balanced_sample')
            self.assertEqual(diagnostic['n_treatment'], 500)
            self.assertEqual(diagnostic['n_control'], 500)
            self.assertEqual(fingerprint(path), before)


if __name__ == '__main__':
    unittest.main()
