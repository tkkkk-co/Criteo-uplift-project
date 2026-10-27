# 项目测试：验证公开快照和演示。

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
from criteo_uplift.pipeline_io import result_directory, write_split_artifacts
import pandas as pd
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
    def test_split_files_are_written_directly_with_explicit_names(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            frame = pd.DataFrame({
                'row_id': range(6), 'treatment': [0, 1] * 3,
                'conversion': [0, 1, 0, 0, 1, 1], 'visit': [1] * 6,
            })
            parts = {name: frame.iloc[i * 2:(i + 1) * 2]
                     for i, name in enumerate(('train', 'validation', 'test'))}
            (root / "data_preparation").mkdir()
            write_split_artifacts(parts, root / "data_preparation")
            self.assertEqual({p.name for p in root.iterdir()}, {'data_preparation'})
            target = root / 'data_preparation'
            self.assertEqual({p.name for p in target.iterdir()}, {
                'split_summary.csv', 'train_row_ids.csv',
                'validation_row_ids.csv', 'test_row_ids.csv',
            })
            summary = pd.read_csv(target / 'split_summary.csv')
            self.assertEqual(len(summary), 6)
            self.assertEqual(summary.n.sum(), 6)
            for name, part in parts.items():
                self.assertEqual(pd.read_csv(target / f'{name}_row_ids.csv').row_id.tolist(), part.row_id.tolist())

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
