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
    """记录文件的路径、大小、修改时间和分块计算的 SHA256。"""
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
    path = Path(path).resolve()
    if path == DATA_ROOT.resolve() or DATA_ROOT.resolve() in path.parents:
        raise ValueError('data/ 只读：请把输出放到 artifacts/。')
    return path


def write_result(value, path):
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
    """验证样本后序列化为 row_id、float32 特征、int8 标签；不接受非法标签截断。"""
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
