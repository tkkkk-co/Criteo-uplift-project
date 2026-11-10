"""核验数据来源，准备建模样本，并划分训练集、验证集和测试集。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [f'f{i}' for i in range(12)]
LABELS = ['treatment', 'conversion', 'visit', 'exposure']
COLUMNS = FEATURES + LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
DEFAULT_DATA = DATA_ROOT / 'raw/criteo-uplift-v2.1.csv'
AUDIT_CACHE_VERSION = 1


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


def prepare_sample(args, *, root: Path = ROOT, progress=None):
    """核验并复用缓存，返回建模样本、全量审计、文件指纹和扫描结果。"""
    # 扫描和缓存模块引用这里的基础定义，在准备样本时再导入。
    from .diagnostics import SAMPLE_CACHE_VERSION, scan_raw_data
    from .pipeline_io import load_sample, safe_output, save_sample, write_result

    if progress is None:
        progress = lambda message: print(message, flush=True)

    if args.c2st_sample < 20 or args.c2st_sample % 20:
        raise ValueError('C2ST sample_size must be >=20 and divisible by 20')
    if args.permutations < 1:
        raise ValueError('permutations must be positive')

    # 1. 分开定位抽样结果、抽样说明和按原始内容指纹共享的全量审计。
    base = safe_output(Path(root) / 'artifacts')
    base.mkdir(exist_ok=True)
    cache = base / f'cache/sample_{args.sample_size}_seed{args.seed}.npz'
    metadata = cache.with_suffix('.json')
    current = fingerprint(args.data)
    audit_cache = base / f'cache/full_audit_{current["sha256"]}.json'

    # 2. 抽样缓存依赖：原始数据、抽样规模、seed 和抽样规则版本。
    reuse_sample = False
    if cache.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        report_fields = {
            'scope', 'sample_cache_version', 'source', 'source_n', 'sample_size',
            'sample_n', 'seed', 'sampling', 'feature_representation',
            'n_treatment', 'n_control',
            'group_summary',
        }
        reuse_sample = (
            report_fields.issubset(info)
            and info.get('scope') == 'modeling_sample'
            and info.get('source', {}).get('sha256') == current['sha256']
            and info.get('sample_size') == args.sample_size
            and info.get('seed') == args.seed
            and info.get('sample_cache_version') == SAMPLE_CACHE_VERSION
        )

    # 3. 全量审计依赖：原始内容、审计规则版本。
    reuse_audit = False
    if audit_cache.exists():
        saved_audit = json.loads(audit_cache.read_text())
        reuse_audit = (
            saved_audit.get('source', {}).get('sha256') == current['sha256']
            and saved_audit.get('audit_cache_version') == AUDIT_CACHE_VERSION
            and isinstance(saved_audit.get('audit'), dict)
            and {'group_summary', 'duplicate_row_fraction', 'duplicate_fraction_denominator',
                 'duplicate_count_definition', 'duplicate_method'}.issubset(saved_audit['audit'])
        )

    # 4. 调用scan_raw_data() 完成全量审计、抽样（建模抽样、C2ST抽样）、诊断（SRM SMD KS）
    progress(
        '统一扫描：全量检查 + C2ST 抽样；'
        + ('复用建模缓存' if reuse_sample else '重新建模抽样') + '；'
        + ('复用全量审计' if reuse_audit else '重新全量审计')
    )
    scanned = scan_raw_data(
        args.data,
        sample_size=None if reuse_sample else args.sample_size,
        seed=args.seed,
        per_arm=args.c2st_sample // 2,
        audit_enabled=not reuse_audit,
        diagnostics_enabled=True,
        progress=progress,
        source=current,
    )

    # 5. 全量审计：复用与重新生成各自集中处理。
    if reuse_audit:
        # 5.1 复用审计：读取已核验来源和版本的缓存，并与本次扫描人数核对。
        audit = saved_audit['audit']
        for audit_key, scan_key in (
            ('n_rows', 'n'), ('n_treatment', 'n_treatment'), ('n_control', 'n_control'),
        ):
            if audit[audit_key] != scanned['metadata'][scan_key]:
                raise ValueError(f'全量审计缓存与当前扫描不一致：{audit_key}')
    else:
        # 5.2 重新审计：取得本次扫描生成的完整审计，并保存独立审计缓存。
        audit = scanned['audit']
        write_result(
            {'source': current, 'audit_cache_version': AUDIT_CACHE_VERSION, 'audit': audit},
            audit_cache,
        )

    # 6. 建模样本：复用与重新抽样各自集中处理，报告由扫描函数生成。
    expected_n = audit['n_rows'] if args.sample_size == 0 else min(args.sample_size, audit['n_rows'])
    if reuse_sample:
        # 6.1 复用样本：读取样本及其已有说明，不重新计算抽样报告。
        sample = load_sample(cache)
        sample_report = info
        if len(sample) != expected_n or sample_report['sample_n'] != len(sample):
            raise ValueError('抽样缓存行数与配置及原始数据规模不一致')
        # 仅统一旧汇总标签，不重新抽样或计算已经正确的统计量。
        if any(row['group'] == '总体' for row in sample_report['group_summary']):
            for row in sample_report['group_summary']:
                if row['group'] == '总体':
                    row['group'] = 'Total'
            write_result(sample_report, metadata)
    else:
        # 6.2 重新抽样：直接取得扫描函数生成的样本和说明，一起保存。
        sample = scanned['sample']
        sample_report = scanned['sample_report']
        if len(sample) != expected_n:
            raise ValueError('抽样结果行数与配置及原始数据规模不一致')
        save_sample(sample, cache)
        write_result(sample_report, metadata)

    scanned['sample_report'] = sample_report
    # 诊断结果无需继续持有建模大表或重复持有审计字典。
    scanned.pop('sample')
    scanned.pop('audit')
    return sample, audit, current, scanned


def split_sample(df, outcome='conversion', seed=42):
    """按处理组和结果分层，返回三个数据集、人数汇总和重叠检查。"""
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
