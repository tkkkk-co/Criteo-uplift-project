#-----------------------------------------------------------------------------------------------------------------------
# 定义数据列名、路径保护、文件指纹、格式转换、缓存读写和数据划分等辅助功能。
# 全量读取、审计、抽样和诊断统一由 diagnostics.py 的 scan_raw_data() 负责。
# 导入本模块不会自动读取原始数据或启动分析。
#-----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [f'f{i}' for i in range(12)]
LABELS = ['treatment', 'conversion', 'visit', 'exposure']
COLUMNS = FEATURES + LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
DEFAULT_DATA = DATA_ROOT / 'raw/criteo-uplift-v2.1.csv'


#-----------------------------------------------------------------------------------------------------------------------
# 函数：safe_output —— 检查结果保存路径，保护原始数据目录。
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
# 函数：write_json —— 将数据保存为 JSON 文件。
# 输入：
#     value：需要保存的 Python 对象。
#     path：目标 JSON 文件路径。
# 输出：
#     返回值：None。
# 保存文件：
#     path：转换后保存的 JSON 文件。
# 说明：检查保存路径、创建父目录，将 NumPy 数值和路径等转换为 JSON 可保存的形式后写入。
#-----------------------------------------------------------------------------------------------------------------------
def write_json(value, path):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：convert —— 转换 JSON 保存过程中不直接支持的对象。
    # 输入：
    #     v：需要转换的对象或其中一个元素。
    # 输出：
    #     转换结果：处理后的对象，返回给 write_json() 进行保存。
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
    path.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False))


#-----------------------------------------------------------------------------------------------------------------------
# 函数：fingerprint —— 记录文件来源信息和内容指纹。
# 输入：
#     path：需要读取的文件路径。
# 输出：字典，各字段如下。
#     path：文件的绝对路径字符串。
#     size：文件大小，单位为字节。
#     mtime_ns：文件修改时间，单位为纳秒。
#     sha256：文件内容的 SHA256 指纹。
# 说明：分块读取整个文件并计算指纹，供其他流程检查数据来源和文件是否发生变化。
#-----------------------------------------------------------------------------------------------------------------------
def fingerprint(path):
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：save_sample —— 将样本保存为压缩缓存。
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
# 函数：load_sample —— 读取已有样本缓存。
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
# 函数：split_sample —— 将样本划分为训练集、验证集和测试集。
# 输入：
#     df：包含 treatment 和结果标签的样本数据表。
#     outcome：用于分层的结果标签，可选 conversion 或 visit，默认 conversion。
#     seed：随机种子，默认 42。
# 输出：字典，各字段如下。
#     train：约占总样本 60% 的训练数据表。
#     validation：约占总样本 20% 的验证数据表。
#     test：约占总样本 20% 的测试数据表。
# 说明：按分组与结果标签的组合分层划分，使三部分尽量保持原样本中的组合比例。
#       三部分使用互不重叠的数据行；模型选择使用验证集，测试集用于最终评价。
#-----------------------------------------------------------------------------------------------------------------------
def split_sample(df, outcome='conversion', seed=42):
    if outcome not in ['conversion', 'visit']:
        raise ValueError('unknown outcome')
    idx = np.arange(len(df))
    strata = df.treatment.to_numpy() * 2 + df[outcome].to_numpy()
    train, holdout = train_test_split(
        idx,
        test_size=0.4,
        stratify=strata,
        random_state=seed,
    )
    val, test = train_test_split(
        holdout,
        test_size=0.5,
        stratify=strata[holdout],
        random_state=seed + 1,
    )
    return {
        name: df.iloc[pos].reset_index(drop=True)
        for name, pos in [('train', train), ('validation', val), ('test', test)]
    }
