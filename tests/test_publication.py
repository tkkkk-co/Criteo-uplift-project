# 项目测试：验证公开快照和演示。

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import ast
import json
import unittest
import numpy as np
from criteo_uplift.pipeline_io import result_directory
from criteo_uplift.config import configure_runtime

configure_runtime()


class PublicationTests(unittest.TestCase):
    def test_learning_notebook_uses_project_functions_and_does_not_train(self):
        """
        函数名：PublicationTests.test_learning_notebook_uses_project_functions_and_does_not_train
        功能：验证学习 Notebook 调用项目正式函数读取结果，不在 Notebook 中重复训练模型。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
        root = Path(__file__).resolve().parents[1]
        notebooks = list((root / 'notebooks').glob('*.ipynb'))
        self.assertEqual([p.name for p in notebooks], ['criteo_uplift_project.ipynb'])
        notebook = json.loads(notebooks[0].read_text())
        forbidden = {'fit', 'fit_transform', '_scan_raw_data', 'prepare_sample',
                     'write_result', 'to_csv', 'write_text', 'write_bytes', 'savefig'}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
                    self.assertNotIn(name, forbidden)
            self.assertFalse(any(o.get('output_type') == 'error' for o in cell.get('outputs', [])))
        text = notebooks[0].read_text()
        for section in ('Stability', 'Score Policy', 'Planning', 'ATE', 'S / T / X'):
            self.assertIn(section, text)

    def test_local_priority_and_explicit_snapshot_mode(self):
        """
        函数名：PublicationTests.test_local_priority_and_explicit_snapshot_mode
        功能：验证结果目录自动模式优先读取本地 artifacts，且 published 模式只读取公开快照。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
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

    def test_invalid_paths_and_modes(self):
        """
        函数名：PublicationTests.test_invalid_paths_and_modes
        功能：验证结果目录定位拒绝路径穿越、绝对路径和不支持的结果模式。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
        for name in ['', '..', '../data', '/tmp/test', 'a/b']:
            with self.assertRaises(ValueError):
                result_directory(name)
        with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'typo'}):
            with self.assertRaises(ValueError):
                result_directory('run')
        with TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                result_directory('absent', d)
