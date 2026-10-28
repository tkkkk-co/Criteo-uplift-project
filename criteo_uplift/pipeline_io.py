# 统一管理路径保护、JSON、样本缓存和实验产物读写。
# 这里负责“怎么保存/读取”，不负责决定何时训练或评价；主流程见 run_pipeline.py。
# Notebook 与结果文件映射见 docs/RESULTS_GUIDE.md。

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from .data import DATA_ROOT, FEATURES, LABELS, ROOT


#-----------------------------------------------------------------------------------------------------------------------
# 函数：safe_output() —— 检查结果保存路径，保护原始数据目录。
# 输入：
#     path：准备写入结果的文件或目录路径。
# 输出：
#     path：解析后的绝对路径对象。
# 说明：禁止向项目 data 目录及其内部写入结果，符号链接指向该目录时也会被识别。
#-----------------------------------------------------------------------------------------------------------------------
def safe_output(path):
    path = Path(path).resolve()
    if path == DATA_ROOT.resolve() or DATA_ROOT.resolve() in path.parents:
        raise ValueError('data/ 只读：请把输出放到 artifacts/。')
    return path


#-----------------------------------------------------------------------------------------------------------------------
# 函数：write_result() —— 按文件后缀统一保存已经准备好的结果。
# 输入：
#     value：JSON 使用字典或列表；CSV 使用 pandas DataFrame。
#     path：完整目标路径，支持 .json 和 .csv 后缀。
# 输出：
#     返回值：None。
# 保存文件：
#     path：保存的 JSON 或 CSV 文件；CSV 不保存 DataFrame 索引。
# 说明：只做类型检查、序列化和保存，不计算统计量、不推断业务目录。
#       保留原始数据路径保护；后缀或类型不匹配时，在写入前报错。
#-----------------------------------------------------------------------------------------------------------------------
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
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：convert —— 转换 JSON 保存过程中不直接支持的对象。
    # 输入：
    #     v：需要转换的对象或其中一个元素。
    # 输出：
    #     转换结果：处理后的对象，返回给 write_result() 进行保存。
    # 说明：递归处理字典和序列，将 NumPy 数组及数值转换为普通 Python 类型，
    #       将路径转换为字符串，将非有限浮点数转换为 None。
    #-----------------------------------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：save_sample() —— 将样本保存为压缩缓存。
# 输入：
#     sample：包含原始行号、特征和标签的样本数据表。
#     path：目标缓存路径，通常使用 .npz 后缀。
# 输出：
#     返回值：None。
# 保存文件：压缩 NPZ 文件，各数组如下。
#     row_id：样本对应的原始 CSV 行号。
#     X：float32 类型的特征矩阵。
#     labels：int8 类型的标签矩阵。
# 说明：检查保存路径并创建父目录，将样本压缩保存，供后续运行直接读取。
#-----------------------------------------------------------------------------------------------------------------------
def save_sample(sample, path):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_id=sample.row_id.to_numpy(),
        X=sample[FEATURES].to_numpy(dtype=np.float32),
        labels=sample[LABELS].to_numpy(dtype=np.int8),
    )


#-----------------------------------------------------------------------------------------------------------------------
# 函数：load_sample() —— 读取已有样本缓存。
# 输入：
#     path：之前保存的 NPZ 缓存路径。
# 输出：
#     frame：恢复后的样本数据表，包含原始行号、特征和标签。
# 说明：从缓存中读取各个数组，恢复列名并组合成数据表；缓存来源由上层流程负责核对。
#-----------------------------------------------------------------------------------------------------------------------
def load_sample(path):
    with np.load(path, allow_pickle=False) as z:
        frame = pd.DataFrame(z['X'], columns=FEATURES)
        for j, c in enumerate(LABELS):
            frame[c] = z['labels'][:, j]
        frame.insert(0, 'row_id', z['row_id'])
    return frame



#-----------------------------------------------------------------------------------------------------------------------
# 函数：result_directory —— 选择 Notebook 应读取的结果目录。
# 输入：
#     name：需要构造、读取或保存的对象名称。
#     root：项目根目录，用于定位缓存或结果。
# 输出：
#     目录路径：本地实验或已发布快照的位置。
# 说明：默认优先使用本地实验结果，找不到时读取公开聚合快照；显式快照模式只读取公开结果。
#-----------------------------------------------------------------------------------------------------------------------
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
