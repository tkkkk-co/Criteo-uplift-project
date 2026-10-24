"""用独立合成数据验证真实 CLI 对应流程，不依赖或更改原始 Criteo。"""

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


class EndToEndTests(unittest.TestCase):
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
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertTrue(manifest['selection']['frozen_before_test'])
            val = pd.read_csv(out / 'validation_policies.csv')
            winner = (
                val.sort_values('incremental_per_100k', ascending=False, kind='stable')
                .iloc[0]
                .policy
            )
            self.assertEqual(manifest['selection']['selected_policy'], winner)
            ranking = pd.read_csv(out / 'test_ranking.csv')
            self.assertIn('qini_score_traditional', ranking)
            policies = pd.read_csv(out / 'test_policies.csv')
            self.assertEqual(
                policies[policies.budget == 1].incremental_per_100k.nunique(), 1
            )
            self.assertTrue((out / 'report.html').stat().st_size > 1000)
            self.assertEqual(len(list((out / 'figures').glob('*.png'))), 5)
            balance = pd.read_csv(out / 'balance.csv')
            self.assertTrue(balance['scope'].eq('full_published_dataset').all())
            self.assertTrue((balance.n_treatment + balance.n_control == len(frame)).all())
            diagnostic = json.loads((out / 'c2st_details.json').read_text())
            self.assertEqual(diagnostic['scope'], 'full_source_arm_balanced_sample')
            self.assertEqual(diagnostic['n_treatment'], 500)
            self.assertEqual(diagnostic['n_control'], 500)
            self.assertEqual(fingerprint(path), before)


if __name__ == '__main__':
    unittest.main()
