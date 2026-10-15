"""Locate local results or lightweight published aggregate snapshots."""

from pathlib import Path
import os

from .data import ROOT

# result_directory() 函数的作用是根据实验名称，定位并返回已经保存的实验结果目录。
# 函数首先检查传入的实验名称是否合法，并将项目根目录转换为 Path 对象；
# 随后读取环境变量 CRITEO_RESULTS_MODE，确定结果查找模式：如果是默认的 auto 模式，则优先查找本地 artifacts/ 目录，若不存在则查找 reports/published/ 中的历史结果快照；如果是 published 模式，则只查找已发布的结果快照。
# 函数通过 for 循环依次检查候选目录是否存在，找到后立即使用 return 返回对应路径；如果所有候选目录都不存在，则抛出 FileNotFoundError。
# 该函数本质上是一个实验结果路径管理工具，不负责数据分析或模型训练，也不会自动生成缺失的实验结果，主要用于支持 Notebook 读取已有实验结果，避免不必要的重复计算，并明确实验结果的来源。


def result_directory(name, root=ROOT):
    """CRITEO_RESULTS_MODE=published forces snapshot-only notebook execution.

    Otherwise prefer a local run and fall back to reports/published. This
    helper is for reading presentation results; it never synthesizes or
    silently re-trains missing experiments.
    """
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name in ('.', '..')
    ):
        raise ValueError('name must be a single result-directory name')
    root = Path(root)
    mode = os.environ.get('CRITEO_RESULTS_MODE', 'auto')
    if mode not in ('auto', 'published'):
        raise ValueError('CRITEO_RESULTS_MODE must be auto or published')
    candidates = [root / 'reports/published' / name]
    if mode == 'auto':
        candidates.insert(0, root / 'artifacts' / name)
    for directory in candidates:
        if directory.is_dir():
            return directory
    raise FileNotFoundError(f'No saved results for {name}; see docs/REPRODUCIBILITY.md')
