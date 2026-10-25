# 统一管理路径保护、JSON、文件指纹、样本缓存和实验产物读写。

from pathlib import Path
import importlib.metadata
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from .data import DATA_ROOT, FEATURES, LABELS, ROOT
from .evaluation import aggregate_ate_from_audit


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
# 函数：write_json() —— 将数据保存为 JSON 文件。
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
# 函数：fingerprint() —— 记录文件来源信息和内容指纹。
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
# 函数：write_split_artifacts —— 保存各集合的原始行号和分组统计。
# 输入：
#     parts：按 train、validation、test 命名的数据划分字典。
#     out：当前实验的结果保存目录。
# 输出：
#     返回值：None；保存 splits.csv 和各集合的行号文件。
# 说明：保存各集合的原始行号和分组统计。
#-----------------------------------------------------------------------------------------------------------------------
def write_split_artifacts(parts, out):
    out = safe_output(out)
    split_summary = []
    for name, frame in parts.items():
        frame[["row_id"]].to_csv(out / f"{name}_row_ids.csv", index=False)
        for arm in (0, 1):
            rows = frame[frame.treatment == arm]
            split_summary.append(
                {
                    "split": name,
                    "treatment": arm,
                    "n": len(rows),
                    "conversion_positive": int(rows.conversion.sum()),
                    "visit_positive": int(rows.visit.sum()),
                }
            )
    pd.DataFrame(split_summary).to_csv(out / "splits.csv", index=False)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：write_feature_overlap —— 检查验证和测试特征是否出现在训练集中。
# 输入：
#     parts：按 train、validation、test 命名的数据划分字典。
#     out：当前实验的结果保存目录。
# 输出：
#     overlap：验证集、测试集匹配训练特征的行比例字典。
# 说明：只比较匿名特征向量，不能把相同特征直接解释为同一用户。
#-----------------------------------------------------------------------------------------------------------------------
def write_feature_overlap(parts, out):
    train = parts["train"]
    validation = parts["validation"]
    test = parts["test"]
    train_hash = pd.util.hash_pandas_object(
        train[FEATURES], index=False
    ).to_numpy()
    overlap = {
        name: float(
            np.isin(
                pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy(),
                train_hash,
            ).mean()
        )
        for name, frame in (("validation", validation), ("test", test))
    }
    write_json(
        {
            "note": (
                "Identical anonymized feature vectors, not identified users; "
                "based on modeling float32 representation"
            ),
            "fraction_matching_training_features": overlap,
        },
        Path(out) / "feature_overlap.json",
    )
    return overlap


#-----------------------------------------------------------------------------------------------------------------------
# 函数：write_audit_artifacts —— 保存全量审计及两个结果变量的平均效果。
# 输入：
#     audit：全量审计的聚合统计字典。
#     out：当前实验的结果保存目录。
# 输出：
#     返回值：None；保存 audit.json 和 ate_full.csv。
# 说明：保存全量审计及两个结果变量的平均效果。
#-----------------------------------------------------------------------------------------------------------------------
def write_audit_artifacts(audit, out):
    out = Path(out)
    write_json(audit, out / "audit.json")
    pd.DataFrame(
        [aggregate_ate_from_audit(audit, outcome) for outcome in ("conversion", "visit")]
    ).to_csv(out / "ate_full.csv", index=False)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：write_run_manifest —— 保存本次运行的配置、来源和环境记录。
# 输入：
#     out：当前实验的结果保存目录。
#     args：本次运行的配置对象，包含数据路径、规模、种子和输出设置。
#     assignment_probability：固定使用的处理分配概率估计值。
#     selection：在验证集上冻结的策略选择记录。
#     model_metadata：模型参数和训练耗时记录。
#     source：输入来源路径或指纹记录，具体类型由本函数的用途决定。
#     sample_n：本次建模样本的总行数。
#     started_at：任务开始时记录的单调时钟读数。
# 输出：
#     返回值：None；写入等待报告完成的 manifest.json。
# 说明：记录输入来源、运行参数、模型配置和依赖版本，报告生成完成后再标记整体运行成功。
#-----------------------------------------------------------------------------------------------------------------------
def write_run_manifest(
    out,
    *,
    args,
    assignment_probability,
    selection,
    model_metadata,
    source,
    sample_n,
    started_at,
):
    write_json(
        {
            "assignment_probability": assignment_probability,
            "probability_source": "training treatment fraction; plug-in estimate",
            "ci_method": (
                "paired within-arm percentile bootstrap, fixed learned scores and "
                "assignment probability; top-k reselected"
            ),
            "uncertainty_excluded": [
                "model training and selection",
                "assignment probability estimation",
                "unidentified user dependence",
            ],
            "estimand": (
                "public-sample assignment (ITT-style) incremental outcome relative "
                "to treat-none"
            ),
            "cost_assumption": "equal cost/value; budget is contact fraction, not currency",
            "ranking_metric_version": (
                "v2; Qini retains control-only zero prefixes; plug-in AUUC explicit; "
                "no perfect normalization"
            ),
            "primary_outcome": args.outcome,
            "selection": selection,
            "models": model_metadata,
            "assumptions": [
                "released sample has approximately constant assignment probability conditional on X",
                "independent rows, consistency and no interference",
                "equal value per conversion and equal contact cost",
            ],
            "status": "metrics_complete",
            "source": source,
            "features": FEATURES,
            "seed": args.seed,
            "sample_n": sample_n,
            "arguments": vars(args),
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "statsmodels",
                    "scikit-learn",
                    "xgboost",
                    "matplotlib",
                    "joblib",
                )
            },
            "elapsed_seconds": time.perf_counter() - started_at,
        },
        Path(out) / "manifest.json",
    )


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
    raise FileNotFoundError(f'No saved results for {name}; see docs/REPRODUCIBILITY.md')
