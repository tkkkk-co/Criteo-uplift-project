# 项目测试：验证公开快照和演示。

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast
import json
import unittest
import numpy as np
from criteo_uplift.pipeline_io import result_directory
from criteo_uplift.runtime import configure_runtime

configure_runtime()
from criteo_uplift.demo import run_demo


#-----------------------------------------------------------------------------------------------------------------------
# 类：PublicationTests —— 验证公开快照和演示。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证公开快照和演示。
#-----------------------------------------------------------------------------------------------------------------------
class PublicationTests(unittest.TestCase):
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_presentation_notebooks_do_not_compute_or_write_artifacts —— 防止展示页重新承担计算和写入职责。
    # 输入：仓库内 Notebook 01/02 的代码单元格。
    # 输出：断言通过；不运行 Notebook，不写入结果。
    # 说明：禁止函数定义、检验/训练/抽样和文件写入入口；允许读取、选列、格式化和显示。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_presentation_notebooks_do_not_compute_or_write_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = {
            'write_result', 'write_text', 'write_bytes', 'to_csv', 'to_json', 'savefig',
            'fit', 'fit_transform', 'scan_raw_data', 'split_sample', 'prepare_sample',
            'load_sample', 'hist', 'groupby', 'sum', 'mean', 'std', 'var',
            'c2st', 'aggregate_ate_from_audit', 'build_report', 'open',
        }
        for filename in ('01_data_understanding.ipynb', '02_sanity_and_ate.ipynb'):
            notebook = json.loads((root / 'notebooks' / filename).read_text())
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                tree = ast.parse(''.join(cell['source']))
                for node in ast.walk(tree):
                    self.assertNotIsInstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
                    if isinstance(node, ast.Call):
                        name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
                        self.assertNotIn(name, forbidden, (filename, name))
        exporter = (root / 'scripts/export_public_results.py').read_text()
        self.assertNotIn('01_data_understanding.ipynb', exporter)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_local_priority_and_explicit_snapshot_mode —— 核对本地结果优先级及显式公开快照模式。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：核对本地结果优先级及显式公开快照模式。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_local_priority_and_explicit_snapshot_mode(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            local, public = root / 'artifacts/run', root / 'reports/published/run'
            public.mkdir(parents=True)
            with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'auto'}):
                self.assertEqual(result_directory('run', root), public)
                local.mkdir(parents=True)
                self.assertEqual(result_directory('run', root), local)
            with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'published'}):
                self.assertEqual(result_directory('run', root), public)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_invalid_paths_and_modes —— 确认结果定位拒绝非法路径与模式。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认结果定位拒绝非法路径与模式。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_invalid_paths_and_modes(self):
        for name in ['', '..', '../data', '/tmp/test', 'a/b']:
            with self.assertRaises(ValueError):
                result_directory(name)
        with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'typo'}):
            with self.assertRaises(ValueError):
                result_directory('run')
        with TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                result_directory('absent', d)

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：test_synthetic_demo_real_learners_freeze_selection_and_no_overwrite —— 确认合成演示使用真实学习器、冻结选择并拒绝覆盖。
    # 输入：
    #     self：当前对象，保存模型状态或测试上下文。
    # 输出：
    #     返回值：None；通过断言验证预期行为，失败时报告错误。
    # 说明：确认合成演示使用真实学习器、冻结选择并拒绝覆盖。
    #-----------------------------------------------------------------------------------------------------------------------
    def test_synthetic_demo_real_learners_freeze_selection_and_no_overwrite(self):
        with TemporaryDirectory() as d:
            out = Path(d) / 'demo'
            policies = run_demo(out, n=1600, trees=3, bootstrap=5)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertFalse(manifest['is_criteo_result'])
            self.assertEqual(manifest['dataset'], 'synthetic randomized demonstration')
            self.assertTrue(manifest['selection']['frozen_before_test'])
            self.assertEqual(len(policies), 6)
            self.assertTrue(np.isfinite(policies.incremental_per_100k).all())
            self.assertEqual(int(policies.n_selected.iloc[0]), 64)
            with self.assertRaises(FileExistsError):
                run_demo(out, n=1600)
