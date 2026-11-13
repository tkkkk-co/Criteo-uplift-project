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
            self.assertEqual(len(test_ranking.call_args.args[2]), 5)
            self.assertEqual(manifest['selection']['selection_statistic'], 'bootstrap_mean')
            policies = pd.read_csv((out / 'evaluation' / 'test_metrics.csv'))
            self.assertEqual(set(policies.budget), {.02, .05, .1, .2, .3, .5, 1.})
            np.testing.assert_array_equal(policies.n_selected, np.floor(policies.budget * summary.loc[summary.split == 'test', 'n'].sum()))
            self.assertEqual(
                policies[policies.budget == 1].gain_ht_per_100k.nunique(), 1
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
            self.assertEqual(path.read_bytes(), before)
            stability = json.loads((out / 'stability/report.json').read_text())
            self.assertEqual(stability['experiments'], 12)
            self.assertTrue(stability['all_test_row_overlaps_zero'])
            self.assertTrue(stability['all_group_split_feature_overlaps_zero'])
            self.assertTrue(json.loads((out / 'business/assumptions.json').read_text())['hypothetical'])
            html = (out / 'report.html').read_text()
            # 公式必须包含可离线显示的矢量路径，不能退回未排版的代码文本或依赖外部脚本。
            equations = re.findall(r'class="math-equation" src="data:image/svg\+xml;base64,([^"]+)"', html)
            self.assertEqual(len(equations), 16)
            for encoded in equations:
                svg = ElementTree.fromstring(base64.b64decode(encoded))
                self.assertTrue(svg.findall('.//{http://www.w3.org/2000/svg}path'))
            self.assertNotIn('<script', html)
            business = re.search(r'<section id="business">(.+?)</section>', html, re.S).group(1)
            self.assertEqual(re.findall(r'<h3>([^<]+)</h3>', business), [
                '1.1 广告投放的业务问题',
                '1.2 Response Propensity 与 Uplift 的直观比较',
                '1.3 因果增量与 CATE',
                '1.4 本项目的研究问题与定向策略',
            ])
            self.assertIn('示意数据，非 Criteo 实验结果。', business)
            self.assertIn('class="targeting-example"', business)
            for value in ('80%', '85%', '+5 pp', '10%', '35%', '+25 pp'):
                self.assertIn(f'<td>{value}</td>', business)
            self.assertIn('平均处理效应（ATE）', business)
            self.assertIn('条件平均处理效应（CATE）', business)
            self.assertIn('各自横轴尺度', html)
            self.assertIn(f'conversion 正例 {int(summary.loc[summary.split == "test", "conversion_positive"].sum()):,} 个', html)
            self.assertIn('<h3>扩大覆盖的边际增量</h3>', html)
            self.assertIn('<h3>全量投放情景</h3>', html)
            self.assertIn('<h3>开发集稳定性分析</h3>', html)
            self.assertIn('未计算边际增量 CI', html)
            self.assertNotRegex(html, r'<td>\s*(?:NaN|nan|None)\s*</td>')
            # 从最终 HTML 反查表格，独立验证展示缩放和新增区间贡献。
            tables = {}
            for section_id in ('ranking-metrics', 'budget-sensitivity', 'marginal-gain'):
                markup = re.search(rf'id="{section_id}">.*?(<table.*?</table>)', html, re.S).group(1)
                table = ElementTree.fromstring(markup)
                headers = [cell.text for cell in table.findall('./thead/tr/th')]
                rows = [[cell.text for cell in row] for row in table.findall('./tbody/tr')]
                self.assertTrue(rows)
                tables[section_id] = pd.DataFrame(rows, columns=headers)
            shown_ranking = tables['ranking-metrics'].set_index('政策')
            self.assertEqual(len(shown_ranking.columns), 2)
            for column, label in (('auuc', 'AUUC'), ('qini_score', 'Traditional Qini Score')):
                for row in ranking.drop_duplicates("policy").itertuples():
                    self.assertEqual(shown_ranking.loc[row.policy, f'{label} × 10⁵'], f'{getattr(row, column) * 100_000:,.2f}')
            shown_budgets = tables['budget-sensitivity']
            self.assertEqual(set(shown_budgets['投放覆盖率']), {'10%', '20%', '30%', '100%'})
            self.assertEqual(len(shown_budgets), 4 * len({winner, 'random', 'propensity'}))
            frozen = policies[policies.policy == winner].sort_values('budget')
            shown_marginal = tables['marginal-gain']
            expected_marginal = np.diff(np.r_[0., frozen.gain_ht_per_100k])
            self.assertEqual(shown_marginal.iloc[:, -1].tolist(), [f'{v:,.2f}' for v in expected_marginal])
            self.assertEqual(shown_marginal['新增人数'].astype(int).tolist(), np.diff(np.r_[0, frozen.n_selected]).tolist())
            for encoded in re.findall(r'src="data:image/png;base64,([^"]+)"', html):
                self.assertTrue(base64.b64decode(encoded).startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertIn('<h3>训练集覆盖与正值性</h3>', html)
            self.assertIn('不能证明高维联合支持', html)
            self.assertIn('C2ST 置换 AUC 分布', html)
            encoded_c2st = base64.b64encode((out / 'figures/c2st_null.png').read_bytes()).decode()
            self.assertIn(encoded_c2st, html)
            self.assertFalse((out / 'diagnostics/ate_full.csv').exists())
            self.assertFalse((out / 'evaluation/ate_test.json').exists())
            # 公开路径只复制聚合结果；完整实验中的模型、逐行预测和额外文件不进入公开目录。
            self.assertNotIn(str(root), html)
            self.assertNotIn('predictions.npz', html)
            self.assertNotIn('data_preparation/test_row_ids.csv', html)
            self.assertEqual(policies.loc[policies.budget.eq(1.), 'gain_ht'].nunique(), 1)
            expected_chapters = [
                ("summary", "00. 执行摘要"),
                ("business", "01. 业务问题与定向策略"),
                ("data", "02. 数据来源与准备"),
                ("diagnostics", "03. 实验诊断与因果识别"),
                ("ate", "04. 平均处理效应（ATE）"),
                ("modeling", "05. Uplift 建模"),
                ("evaluation", "06. 评估框架"),
                ("validation", "07. 验证与模型选择"),
                ("test", "08. 独立测试评估"),
                ("robustness", "09. 预算敏感性与稳健性"),
                ("conclusion", "10. 业务启示与结论"),
                ("appendix", "附录：技术细节与复现"),
            ]
            chapters = re.findall(r'<section[^>]* id="([^"]+)"><h2>([^<]+)</h2>', html)
            navigation = re.findall(r'<li><a href="#([^"]+)">([^<]+)</a></li>', html)
            self.assertEqual(chapters, expected_chapters)
            self.assertEqual(navigation, expected_chapters)
            with self.assertRaises(FileExistsError):
                run(args)




if __name__ == '__main__':
    unittest.main()
