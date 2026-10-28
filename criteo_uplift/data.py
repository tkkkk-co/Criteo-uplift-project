#-----------------------------------------------------------------------------------------------------------------------
# 定义数据列名、项目路径常量、文件指纹和训练、验证、测试集划分。
# 路径保护、JSON 和样本缓存读写统一放在 pipeline_io.py。
# 全量读取、审计、抽样和诊断统一由 diagnostics.py 的 scan_raw_data() 负责。
# 导入本模块不会自动读取原始数据或启动分析。
#-----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
import hashlib
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
# 函数：split_sample() —— 划分样本、核验来源行、生成汇总和特征重叠报告。
# 输入：
#     df：待划分的建模样本，包含唯一非空的 row_id、FEATURES、treatment 和分层结果列。
#     outcome：用于分层的结果标签，可选 conversion 或 visit，默认 conversion。
#     seed：随机种子，默认 42。
# 输出：
#     parts：包含 train、validation、test 的字典，约占样本 60%、20%、20%。
#     split_summary：各集合按 treatment 分组的人数及正例数表。
#                    split、treatment、n：集合名、处理臂和行数。
#                    conversion_positive、visit_positive：输入中存在相应结果列时统计正例数。
#     feature_overlap：来源行核验结果和三对集合的特征匹配比例、字段与精度说明。
# 说明：按分组与结果标签的组合分层划分，使三部分尽量保持原样本中的组合比例。
#       三部分使用互不重叠的数据行；模型选择使用验证集，测试集用于最终评价。
#       每次划分均核验 row_id 的唯一性、集合互斥与完整覆盖，不复用其他划分的检查结果。
#       row_id 只标识来源行，不能证明用户级隔离；特征相同仅作为诊断，不报错或删行。
#       本函数只计算并返回结果；文件保存由调用方安排。
#-----------------------------------------------------------------------------------------------------------------------
def split_sample(df, outcome='conversion', seed=42):
    if outcome not in ['conversion', 'visit']:
        raise ValueError('unknown outcome')
    # 1. 保留来源行标识；不能用划分后重新编号的 DataFrame 索引代替。
    if 'row_id' not in df.columns:
        raise ValueError('划分样本缺少来源行标识：row_id')
    source_ids = pd.Index(df['row_id'])
    if source_ids.hasnans or not source_ids.is_unique:
        raise ValueError('划分前的 row_id 必须非空且唯一')

    # 2. 按 treatment × outcome 分层，划分规则和随机种子保持不变。
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
    parts = {
        name: df.iloc[pos].reset_index(drop=True)
        for name, pos in [('train', train), ('validation', val), ('test', test)]
    }

    # 3. 检查本次实际划分：集合内部唯一、集合之间互斥，且完整覆盖输入样本。
    split_ids = {name: pd.Index(frame['row_id']) for name, frame in parts.items()}
    names = list(split_ids)
    for i, name in enumerate(names):
        identifiers = split_ids[name]
        if identifiers.hasnans or not identifiers.is_unique: #集合内部缺失 重复
            raise ValueError(f'{name} 的 row_id 必须非空且唯一')
        for other in names[i + 1:]:
            overlap = identifiers.intersection(split_ids[other])
            if len(overlap):
                raise ValueError(f'{name} 与 {other} 存在重复来源行：{len(overlap)} 个 row_id')
            
    combined_ids = split_ids['train'].append([split_ids['validation'], split_ids['test']])
    if len(combined_ids) != len(source_ids) or not source_ids.isin(combined_ids).all():
        raise ValueError('划分后的 row_id 未完整覆盖原样本，或混入了其他来源行')

    # 4. 汇总各集合和处理臂的人数及输入中已有结果列的正例数。
    summary = []
    for name, frame in parts.items():
        for arm in (0, 1):
            rows = frame[frame.treatment == arm]
            summary.append(
                {
                    "split": name,
                    "treatment": arm,
                    "n": len(rows),
                    **{
                        f"{label}_positive": int(rows[label].sum())
                        for label in ("conversion", "visit")
                        if label in df.columns
                    },
                }
            )
    split_summary = pd.DataFrame(summary)

    # 5. 按传入样本的实际特征精度筛查内容重叠，不改变特征或使用结果标签。
    #    每条被检查行只计一次，分母是被检查集合的行数；哈希匹配不是用户身份核验。
    feature_hashes = {
        name: pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()
        for name, frame in parts.items()
    }
    feature_overlap = {
        "note": (
            "基于输入样本的实际特征表示，检查匿名特征向量的哈希匹配；"
            "分母为被检查集合的行数，特征相同不代表属于同一用户。"
            "row_id 核验通过仅说明来源行划分有效，不能证明用户级隔离。"
        ),
        "features": FEATURES.copy(),
        "feature_dtypes": {feature: str(df[feature].dtype) for feature in FEATURES},
        "row_id_checks": {
            "non_null_unique": True,
            "pairwise_disjoint": True,
            "complete_coverage": True,
        },
        "fraction_matching_training_features": {
            name: float(np.isin(feature_hashes[name], feature_hashes['train']).mean())
            for name in ('validation', 'test')
        },
        "fraction_test_matching_validation_features": float(
            np.isin(feature_hashes['test'], feature_hashes['validation']).mean()
        ),
    }
    return parts, split_summary, feature_overlap
