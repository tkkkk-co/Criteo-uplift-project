# 项目测试：验证公开快照和演示。

from pathlib import Path
import ast
import json
import unittest
from criteo_uplift.config import configure_runtime

configure_runtime()


class PublicationTests(unittest.TestCase):
    def test_learning_notebook_uses_project_functions_and_does_not_train(self):
        #-------------
        # 函数名：PublicationTests.test_learning_notebook_uses_project_functions_and_does_not_train
        # 功能：验证学习 Notebook 调用项目正式函数读取结果，不在 Notebook 中重复训练模型。
        # 输入参数：
        #     - 无显式业务参数（实例方法中的 self 表示当前对象）。
        # 输出参数：
        #     - 无；通过断言验证预期行为，失败时由测试框架报告。
        #-------------
        root = Path(__file__).resolve().parents[1]
        notebooks = list((root / 'notebooks').glob('*.ipynb'))
        self.assertEqual([p.name for p in notebooks], ['criteo_uplift_project.ipynb'])
        notebook = json.loads(notebooks[0].read_text())
        forbidden = {'fit', 'fit_transform', 'scan_raw_data', 'prepare_sample',
                     'write_result', 'to_csv', 'write_text', 'write_bytes', 'savefig'}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                self.assertNotIsInstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                if isinstance(node, ast.Call):
                    name = getattr(node.func, 'id', getattr(node.func, 'attr', None))
                    self.assertNotIn(name, forbidden)
            self.assertFalse(any(o.get('output_type') == 'error' for o in cell.get('outputs', [])))
        text = notebooks[0].read_text()
        for section in ('Stability', 'Planning', 'ATE', 'S / T / X'):
            self.assertIn(section, text)

    def test_public_results_exclude_individual_data(self):
        #-------------
        # 功能：公开展示中不应包含模型、缓存、逐行数据或来源行号列表。
        # 输入参数：
        #     - self：当前测试实例。
        # 输出参数：
        #     - 无；检查公开目录的数据范围。
        #-------------
        public = Path(__file__).resolve().parents[1] / 'reports/published'
        self.assertTrue(list(public.rglob('report.html')))
        for path in public.rglob('*'):
            if not path.is_file():
                continue
            self.assertNotIn(path.suffix, ('.npz', '.npy', '.joblib', '.pkl', '.pickle', '.parquet'))
            self.assertNotIn('row_ids', path.name)
            if path.suffix == '.csv':
                import pandas as pd
                columns = set(pd.read_csv(path, nrows=0).columns)
                self.assertFalse(columns & {'row_id', 'source_row_id'})
                self.assertFalse(set(f'f{i}' for i in range(12)).issubset(columns))
