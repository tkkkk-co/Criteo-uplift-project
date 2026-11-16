# 项目测试：验证完整实验流程。

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import base64
import hashlib
import json
import re
import unittest
from xml.etree import ElementTree
import numpy as np
import pandas as pd
from criteo_uplift.config import FEATURES
from criteo_uplift.metrics import evaluate_ranking
from run_pipeline import run
from criteo_uplift.diagnostics import diagnose_overlap


#-----------------------------------------------------------------------------------------------------------------------
# 类：EndToEndTests —— 验证完整实验流程。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证完整实验流程。
#-----------------------------------------------------------------------------------------------------------------------
class EndToEndTests(unittest.TestCase):
    def test_complete_pipeline_outputs_and_frozen_selection(self):
        #-------------
        # 函数名：EndToEndTests.test_complete_pipeline_outputs_and_frozen_selection
        # 功能：验证合成数据能够贯通完整 Pipeline，生成预期结果文件，并在 Test 评价前冻结模型和策略选择。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
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
            before = path.read_bytes()
            args = SimpleNamespace(
                data=path,
                sample_size=4000,
                seed=814,
                outcome='conversion',
                primary_budget=0.2,
                primary_metric="gain_ht",
                permutations=2,
                c2st_sample=1000,
                bootstrap=5,
                trees=5,
                output=root / 'run', planning_assumptions=None, stability_seeds=[17, 29, 83],
                stability_fractions=[.25, .5, 1.],
            )
            with patch('run_pipeline.ROOT', root), patch(
                'criteo_uplift.diagnostics.pd.read_csv', wraps=pd.read_csv,
            ) as read_csv, patch(
                'criteo_uplift.diagnostics.diagnose_overlap', wraps=diagnose_overlap,
            ) as overlap_check, patch(
                'criteo_uplift.evaluation.evaluate_ranking', wraps=evaluate_ranking,
            ) as test_ranking:
                out = run(args)
            self.assertEqual(overlap_check.call_count, 1)
            diagnosed_x, diagnosed_t = overlap_check.call_args.args
            train_ids = pd.read_csv(out / 'data_preparation/train_row_ids.csv').row_id.to_numpy()
            np.testing.assert_array_equal(diagnosed_x.to_numpy(), frame.loc[train_ids, FEATURES].to_numpy(dtype=np.float32))
            np.testing.assert_array_equal(diagnosed_t, frame.loc[train_ids, 'treatment'])
            raw_scans = [
                call for call in read_csv.call_args_list
                if 'chunksize' in call.kwargs
            ]
            self.assertEqual(len(raw_scans), 1)
            for name in (
                'data_preparation/audit.json', 'data_preparation/sample_report.json',
                'data_preparation/split_summary.csv', 'data_preparation/test_row_ids.csv',
                'diagnostics/learnability.csv', 'diagnostics/balance.csv',
                'diagnostics/overlap_positivity.json', 'diagnostics/overlap_strata.csv',
                'models/selection.json', 'models/t_xgboost.joblib',
                'evaluation/test_metrics.csv', 'evaluation/test_bootstrap.csv', 'evaluation/test_pairwise.csv',
                'models/validation_metrics.csv', 'models/validation_bootstrap.csv', 'evaluation/predictions.npz',
                'figures/sample_features.png', 'strategy_overlap.json', 'report_provenance.json',
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
            ate_full = pd.read_csv(out / 'evaluation/ate_summary.csv')
            for outcome in ('conversion', 'visit'):
                expected = ate_summary(frame[outcome], frame.treatment)
                actual = ate_full[(ate_full.outcome == outcome) & (ate_full.scope == "full_dataset")].iloc[0]
                for key, value in expected.items():
                    if key in actual.index and isinstance(value, (int, float)):
                        self.assertAlmostEqual(actual[key], value)
                # 全量报告用二项 plug-in 方差；ate_summary 用 ddof=1 的样本方差。
            self.assertFalse((out / 'splits.csv').exists())
            self.assertFalse((out / 'split_summary.csv').exists())
            manifest = json.loads(((out / 'manifest.json')).read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['source']['sha256'], hashlib.sha256(before).hexdigest())
            coverage = json.loads((out / 'diagnostics/overlap_positivity.json').read_text())
            self.assertEqual(coverage['scope'], 'train')
            self.assertEqual(coverage['n'], len(train_ids))
            self.assertEqual(coverage['design_positivity'], 'not_verified')
            audit = json.loads(((out / 'data_preparation' / 'audit.json')).read_text())
            sample_report = json.loads(((out / 'data_preparation' / 'sample_report.json')).read_text())
            self.assertNotIn('audit', sample_report)
            self.assertEqual(sample_report['sample_n'], 4000)
            self.assertEqual(sample_report['seed'], args.seed)
            self.assertEqual(sample_report['source_n'], audit['n_rows'])
            for key in ('sample_n', 'sample_seed', 'sampling'):
                self.assertNotIn(key, audit)

            self.assertTrue(manifest['selection']['frozen_before_test'])
            val = pd.read_csv((out / 'models' / 'validation_bootstrap.csv'))
            winner = (
                val.query("policy != 'random'").sort_values('bootstrap_mean', ascending=False, kind='stable')
                .iloc[0]
                .policy
            )
            self.assertEqual(manifest['selection']['selected_policy'], winner)
            ranking = pd.read_csv((out / 'evaluation' / 'test_metrics.csv'))
            self.assertIn('qini_score', ranking)
            self.assertEqual(test_ranking.call_count, 1)
            self.assertEqual(len(test_ranking.call_args.args[2]), 8)
            self.assertEqual(manifest['selection']['selection_statistic'], 'bootstrap_mean')
            policies = pd.read_csv((out / 'evaluation' / 'test_metrics.csv'))
            self.assertEqual(set(policies.budget), {.02, .05, .1, .2, .3, .5, 1.})
            np.testing.assert_array_equal(policies.n_selected, np.floor(policies.budget * summary.loc[summary.split == 'test', 'n'].sum()))
            self.assertEqual(
                policies[policies.budget == 1].gain_ht_per_100k.nunique(), 1
            )
            self.assertTrue((out / 'report.html').stat().st_size > 1000)
            self.assertEqual(len(list((out / 'figures').glob('*.png'))), 1)
            balance = pd.read_csv((out / 'diagnostics' / 'balance.csv'))
            self.assertTrue(balance['scope'].eq('full_published_dataset').all())
            self.assertTrue((balance.n_treatment + balance.n_control == len(frame)).all())
            diagnostic = json.loads(((out / 'diagnostics' / 'c2st_details.json')).read_text())
            self.assertEqual(diagnostic['scope'], 'full_source_arm_balanced_sample')
            self.assertEqual(diagnostic['n_treatment'], 500)
            self.assertEqual(diagnostic['n_control'], 500)
            self.assertEqual(path.read_bytes(), before)
            stability = json.loads((out / 'stability/report.json').read_text())
            self.assertEqual(stability['experiments'], 12)
            self.assertTrue(stability['all_test_row_overlaps_zero'])
            self.assertTrue(stability['all_group_split_feature_overlaps_zero'])
            self.assertTrue(json.loads((out / 'business/assumptions.json').read_text())['hypothetical'])
            html = (out / 'report.html').read_text()
            equations = re.findall(r'class="math-equation" src="data:image/svg\+xml;base64,([^"]+)"', html)
            self.assertGreaterEqual(len(equations), 12)
            for encoded in equations:
                svg = ElementTree.fromstring(base64.b64decode(encoded))
                self.assertTrue(svg.findall('.//{http://www.w3.org/2000/svg}path'))
            self.assertNotRegex(html, r'<script[^>]+src=')
            body = re.search(r'<article id="research".+?</article>', html, re.S).group()
            self.assertEqual(len(re.findall(r'<section id=', body)), 7)
            self.assertNotIn('<details', body)
            self.assertNotIn('class="table-wrap wide"', body)
            for text in ('RESEARCH PROFILE', 'Criteo Uplift Prediction Dataset v2.1',
                         'Randomized Experiment (RCT)', 'Uplift Modeling',
                         'ATE<sub>T</sub> (ITT) / CATE<sub>T</sub>', 'HT-Gain@20%',
                         '项目研究流程', '示意数据，非 Criteo 实验结果', 'SUTVA', 'Spearman',
                         '线上实验尚未执行', '理论最优性', 'StandardScaler'):
                self.assertIn(text, html)
            self.assertNotIn('reading-entries', html)
            self.assertNotIn('id="workflow"', html)
            self.assertEqual(len(re.findall(r'<nav class="chapter-nav"', html)), 1)
            # Every saved primary point and interval must appear in the nine-policy figure.
            comparison = re.search(r'<figure id="primary-policy-chart".+?</figure>', html, re.S).group()
            bootstrap = pd.read_csv(out / 'evaluation/test_bootstrap.csv').query('budget == 0.2')
            self.assertEqual(len(re.findall(r'class="policy-row\b', comparison)), 9)
            self.assertIn(f'class="policy-row frozen" data-policy="{winner}"', comparison)
            for row in bootstrap.itertuples():
                self.assertIn(f'{row.point_estimate * 100000:,.2f} <small>[{row.ci_low * 100000:,.2f}, {row.ci_high * 100000:,.2f}]</small>', comparison)
            # Audit tables cover every saved validation/test policy-budget and pairwise row.
            appendix = re.search(r'<section id="appendix">.+</section>', html, re.S).group()
            self.assertNotRegex(appendix, r'<details[^>]*\sopen(?:\s|>)')
            results = re.search(r'id="appendix-results".+?id="appendix-pairs"', html, re.S).group()
            tables = re.findall(r'<table.*?</table>', results, re.S)
            val_metric_rows = ElementTree.fromstring(tables[0]).findall('./tbody/tr')
            test_metric_rows = ElementTree.fromstring(tables[-1]).findall('./tbody/tr')
            self.assertEqual(len(val_metric_rows), 9)
            self.assertEqual(len(test_metric_rows), len(policies))
            for shown, row in zip(test_metric_rows, policies.itertuples()):
                values = [cell.text for cell in shown]
                self.assertEqual(values[3], f'{row.gain_ht * 100000:,.2f}')
                self.assertEqual(values[-2], f'{row.auuc * 100000:,.2f}')
                self.assertEqual(values[-1], f'{row.qini_score * 100000:,.2f}')
            pair_markup = re.search(r'id="appendix-pairs".+?id="appendix-budget"',html,re.S).group()
            pair_tables = re.findall(r'<table.*?</table>',pair_markup,re.S)
            for markup, name in zip(pair_tables, ['models/validation_pairwise.csv','evaluation/test_pairwise.csv']):
                self.assertEqual(len(ElementTree.fromstring(markup).findall('./tbody/tr')),len(pd.read_csv(out/name)))
            self.assertNotIn(str(root), html)
            overlap = json.loads((out/'strategy_overlap.json').read_text())
            self.assertEqual(overlap['n'], len(source_by_id.loc[pd.read_csv(out/'data_preparation/test_row_ids.csv').row_id]))
            self.assertEqual(overlap['policy'], manifest['selection']['selected_uplift_model'])
            # Rebuilding elsewhere must neither train/predict nor alter any original artifact.
            from criteo_uplift.reporting import build_report
            original = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in out.rglob('*') if p.is_file()}
            with patch('criteo_uplift.models.s_learner', side_effect=AssertionError('no refit')), \
                 patch('criteo_uplift.evaluation.evaluate_ranking', side_effect=AssertionError('no evaluation')):
                build_report(out/'manifest.json',root/'published/report.html')
            after = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in out.rglob('*') if p.is_file()}
            self.assertEqual(original,after)
            self.assertTrue((root/'published/report_provenance.json').is_file())
            with self.assertRaises(FileExistsError): build_report(out/'manifest.json',out/'report.html')
            with self.assertRaises(FileExistsError): run(args)





if __name__ == '__main__':
    unittest.main()
