# 统一管理路径保护、JSON、样本缓存和实验产物读写。
# 这里负责“怎么保存/读取”，不负责决定何时训练或评价；主流程见 run_pipeline.py。
# Notebook 与结果文件映射见 docs/RESULTS_GUIDE.md。

from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .config import DATA_ROOT, FEATURES, LABELS, ROOT
from .input_checks import check_binary_vector, check_feature_matrix, check_row_ids

def fingerprint(path):
    """
    函数名：fingerprint
    功能：计算文件的路径、大小、修改时间和分块 SHA256，用于核验数据来源及缓存有效性。
    输入参数：
        - path：需要计算指纹的文件路径。
    输出参数：
        - 来源指纹字典：包含绝对路径、文件大小、mtime_ns 和 SHA256。
    """
    path = Path(path)
    h = hashlib.sha256()

    with path.open('rb') as f:  # 以二进制只读模式读取文件。
        for block in iter(
            lambda: f.read(8*1024*1024),
            b''):
            h.update(block)

    return {
        'path':str(path.resolve()),
        'size':path.stat().st_size,
        'mtime_ns':path.stat().st_mtime_ns,
        'sha256':h.hexdigest()
    }



def safe_output(path):
    """
    函数名：safe_output
    功能：解析输出路径并拒绝将任何文件写入项目只读的 data 目录。
    输入参数：
        - path：待检查和规范化的输出路径。
    输出参数：
        - 解析后的 pathlib.Path 输出路径。
    """
    path = Path(path).resolve()
    if path == DATA_ROOT.resolve() or DATA_ROOT.resolve() in path.parents:
        raise ValueError('data/ 只读：请把输出放到 artifacts/。')
    return path


def write_result(value, path):
    """
    函数名：write_result
    功能：按文件后缀将 DataFrame 写为 CSV，或将可序列化对象写为规范化 JSON。
    输入参数：
        - value：需要保存的 DataFrame、字典、列表或元组。
        - path：目标 CSV/JSON 文件路径。
    输出参数：
        - 无；在目标路径创建或写入结果文件。
    """
    path = safe_output(path)
    suffix = path.suffix.lower()
    if suffix not in (".json", ".csv"):
        raise ValueError("write_result 仅支持 .json 和 .csv 文件")
    if suffix == ".csv":
        if not isinstance(value, pd.DataFrame):
            raise TypeError("CSV 结果必须是 pandas DataFrame")
        path.parent.mkdir(parents=True, exist_ok=True)
        value.to_csv(path, index=False)
        return
    if not isinstance(value, (dict, list, tuple)):
        raise TypeError("JSON 结果必须是字典或列表")
    def convert(v):
    """
    函数名：write_result.convert
    功能：递归转换 NumPy、Path 和容器对象，使其可以安全序列化为 JSON。
    输入参数：
        - v：待递归转换的 Python/NumPy 值。
    输出参数：
        - JSON 可序列化的 Python 基础类型、列表或字典。
    """
        if isinstance(v, dict): return {str(k): convert(x) for k,x in v.items()}
        if isinstance(v, (tuple,list)): return [convert(x) for x in v]
        if isinstance(v, np.ndarray): return convert(v.tolist())
        if isinstance(v, np.generic): return convert(v.item())
        if isinstance(v,float) and not np.isfinite(v): return None
        if isinstance(v, Path): return str(v)
        return v
    content = json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_sample(sample, path):
    """
    函数名：save_sample
    功能：校验建模样本的列、特征、row_id 和标签后，将其压缩保存为 NPZ 缓存。
    输入参数：
        - sample：包含 row_id、FEATURES 和 LABELS 的建模样本 DataFrame。
        - path：NPZ 样本缓存目标路径。
    输出参数：
        - 无；在 path 保存压缩样本数组。
    """
    path = safe_output(path)
    required = {"row_id", *FEATURES, *LABELS}
    if not required.issubset(sample.columns):
        raise ValueError(f"Sample is missing columns: {sorted(required - set(sample.columns))}")
    check_feature_matrix(sample[FEATURES], n_features=len(FEATURES))
    check_row_ids(sample.row_id, n_rows=len(sample), require_integer=True)
    for label in LABELS:
        check_binary_vector(sample[label], label, len(sample))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_id=sample.row_id.to_numpy(),
        X=sample[FEATURES].to_numpy(dtype=np.float32),
        labels=sample[LABELS].to_numpy(dtype=np.int8),
    )


def load_sample(path):
    """
    函数名：load_sample
    功能：读取 NPZ 样本缓存，校验数组形状、row_id 和二元标签后恢复为 DataFrame。
    输入参数：
        - path：待读取的 NPZ 样本缓存路径。
    输出参数：
        - 恢复后的样本 DataFrame，包含 row_id、特征和标签列。
    """
    with np.load(path, allow_pickle=False) as z:
        check_feature_matrix(z['X'], n_features=len(FEATURES))
        check_row_ids(z['row_id'], n_rows=len(z['X']), require_integer=True)
        if z['labels'].shape != (len(z['X']), len(LABELS)):
            raise ValueError('Cached label shape does not match feature rows and label columns')
        for j, label in enumerate(LABELS):
            check_binary_vector(z['labels'][:, j], label, len(z['X']))
        frame = pd.DataFrame(z['X'], columns=FEATURES)
        for j, c in enumerate(LABELS):
            frame[c] = z['labels'][:, j]
        frame.insert(0, 'row_id', z['row_id'])
    return frame



def result_directory(name, root=ROOT):
    """
    函数名：result_directory
    功能：按结果模式在本地 artifacts 或已发布快照目录中定位指定实验结果目录。
    输入参数：
        - name：实验结果目录名称，必须是单层目录名。
        - root：项目根目录，默认使用配置中的 ROOT。
    输出参数：
        - 找到的实验目录 pathlib.Path；不存在时抛出 FileNotFoundError。
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
    raise FileNotFoundError(f'No saved results for {name}; see docs/PROJECT_PIPELINE.md and docs/RESULTS_GUIDE.md')
