"""Prepare the modeling sample, audit and diagnostics in one source scan."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES
from .input_checks import check_binary_vector, check_feature_matrix, check_row_ids
from .raw_scan import scan_raw_data


def prepare_sample(args, *, progress=print, chunk_size=250_000):
    """Read the original CSV and prepare all inputs for this run."""
    if args.c2st_sample < 20 or args.c2st_sample % 20:
        raise ValueError('C2ST sample_size 必须大于等于 20 且能被 20 整除')
    from .experiment_protocol import load_protocol
    frozen = getattr(args, "protocol", None)
    meta, excluded = load_protocol(frozen) if frozen else (None, None)
    result = scan_raw_data(
        args.data, sample_size=args.sample_size, seed=args.seed,
        chunk_size=chunk_size, per_arm=args.c2st_sample // 2, progress=progress,
        excluded_row_ids=excluded if getattr(args, "stage", None) == "development" else None,
    )

    if meta:
        if result['source']['sha256'] != meta['source_sha256'] or result['audit']['n_rows'] != meta['source_n']:
            raise ValueError('源数据与预冻结协议不一致')
        result['sample_report']['protocol'] = meta
        if getattr(args, 'stage', None) == 'development':
            result['sample_report']['sampling'] += '; excludes all frozen final Test row IDs'
            if np.isin(result['sample'].row_id, excluded).any():
                raise AssertionError('开发集包含正式 Test')
    return result


def split_sample(df, outcome='conversion', seed=42, *, frozen_test_ids=None):
    #-------------
    # 函数名：split_sample
    # 功能：按 treatment × outcome 分层将建模样本划分为 Train、Validation、Test，并检查 row_id 互斥及特征重叠。
    # 输入参数：
    #     - df：包含 row_id、特征、处理分配和结果标签的建模样本。
    #     - outcome：用于分层的二元结果列名，支持 conversion 或 visit。
    #     - seed：数据划分随机种子。
    # 输出参数：
    #     - 三元组：(train/validation/test DataFrame 字典、分组人数汇总表、row_id 与匿名特征重叠检查字典)。
    #-------------
    if outcome not in ['conversion', 'visit']:
        raise ValueError('不支持该结果变量；请选择 conversion 或 visit')
    # 1. 保留来源行标识；不能用划分后重新编号的 DataFrame 索引代替。
    if 'row_id' not in df.columns:
        raise ValueError('划分样本缺少来源行标识：row_id')
    check_row_ids(df['row_id'], n_rows=len(df))
    check_feature_matrix(df[FEATURES], allow_nan=True, n_features=len(FEATURES))
    check_binary_vector(df.treatment, 'treatment', len(df), require_both=True)
    check_binary_vector(df[outcome], outcome, len(df))
    source_ids = pd.Index(df['row_id'])

    # 2. 按 treatment × outcome 分层，划分规则和随机种子保持不变。
    idx = np.arange(len(df))
    strata = df.treatment.to_numpy() * 2 + df[outcome].to_numpy()
    if frozen_test_ids is None:
        train, holdout = train_test_split(idx, test_size=.4, stratify=strata, random_state=seed)
        val, test = train_test_split(holdout, test_size=.5, stratify=strata[holdout], random_state=seed + 1)
    else:
        test_mask = df.row_id.isin(frozen_test_ids).to_numpy()
        test, development = idx[test_mask], idx[~test_mask]
        if len(test) != len(frozen_test_ids):
            raise ValueError('全量样本未包含全部冻结 Test 行')
        val_n = int(np.ceil(.4 * len(df))) - len(test)
        train, val = train_test_split(development, test_size=val_n,
                                      stratify=strata[development], random_state=seed + 1)
    parts = {
        name: df.iloc[pos].reset_index(drop=True)
        for name, pos in [('train', train), ('validation', val), ('test', test)]
    }

    # 3. 检查本次实际划分：集合内部唯一、集合之间互斥，且完整覆盖输入样本。
    split_ids = {name: pd.Index(frame['row_id']) for name, frame in parts.items()}
    names = list(split_ids)
    for i, name in enumerate(names):
        identifiers = split_ids[name]
        check_row_ids(identifiers, name)
        for other in names[i + 1:]:
            overlap = identifiers.intersection(split_ids[other])
            if len(overlap):
                raise ValueError(f'{name} 与 {other} 存在重复来源行：{len(overlap)} 个 row_id')

    combined_ids = split_ids['train'].append([split_ids['validation'], split_ids['test']])
    if len(combined_ids) != len(source_ids) or not source_ids.isin(combined_ids).all():
        raise ValueError('划分后的 row_id 未完整覆盖原样本，或混入了其他来源行')

    # 4. 汇总各集合和处理臂的人数及输入中已有结果列的正例数。
    for name, proportion in [('train', .6), ('validation', .2), ('test', .2)]:
        if abs(len(parts[name]) - len(df) * proportion) > 1:
            raise ValueError(f'{name} 的样本数量不符合 60/20/20 划分比例')
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
