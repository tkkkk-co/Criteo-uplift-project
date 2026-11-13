"""Protect the raw data directory and write experiment results."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import DATA_ROOT


def safe_output(path):
    #-------------
    # 函数名：safe_output
    # 功能：解析输出路径并拒绝将任何文件写入项目只读的 data 目录。
    # 输入参数：
    #     - path：待检查和规范化的输出路径。
    # 输出参数：
    #     - 解析后的 pathlib.Path 输出路径。
    #-------------
    path = Path(path).resolve()
    if path == DATA_ROOT.resolve() or DATA_ROOT.resolve() in path.parents:
        raise ValueError('data/ 只读：请把输出放到 artifacts/。')
    return path


def write_result(value, path):
    #-------------
    # 函数名：write_result
    # 功能：按文件后缀将 DataFrame 写为 CSV，或将可序列化对象写为规范化 JSON。
    # 输入参数：
    #     - value：需要保存的 DataFrame、字典、列表或元组。
    #     - path：目标 CSV/JSON 文件路径。
    # 输出参数：
    #     - 无；在目标路径创建或写入结果文件。
    #-------------
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
        #-------------
        # 函数名：write_result.convert
        # 功能：递归转换 NumPy、Path 和容器对象，使其可以安全序列化为 JSON。
        # 输入参数：
        #     - v：待递归转换的 Python/NumPy 值。
        # 输出参数：
        #     - JSON 可序列化的 Python 基础类型、列表或字典。
        #-------------
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
