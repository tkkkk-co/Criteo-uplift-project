#-------------
# 统一完成数据准备、复用独立缓存并划分建模样本。
#-------------
import json
from pathlib import Path
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, FEATURES, LABELS
from .input_checks import check_binary_vector, check_feature_matrix, check_row_ids
from .pipeline_io import fingerprint, load_sample, safe_output, save_sample, write_result
from .raw_scan import scan_raw_data, scan_metadata

_BALANCE_COLUMNS = [
    'feature', 'n_treatment', 'n_control', 'mean_treatment', 'mean_control',
    'smd', 'abs_smd', 'ks_statistic', 'ks_pvalue', 'scope', 'representation', 'source_sha256',
]


def _read_cache_manifest(path):
    #-------------
    # 读取缓存说明；文件缺失、内容截断或 JSON 无效时视为缓存未命中。
    # 输入参数：
    #     - path：缓存 JSON 说明文件的路径。
    # 输出参数：
    #     - 缓存说明字典；文件缺失或 JSON 无效时返回 None。
    #-------------
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_sample_cache(path, source, size, seed):
    #-------------
    # 说明与 NPZ 均通过校验才命中；文件损坏、缺列、形状或数据异常均交由统一扫描重建。
    # 在决定扫描任务前完成读取，避免扫描结束后才发现损坏而被迫再次扫描来源。
    # 输入参数：
    #     - path：待校验并读取的建模样本 NPZ 路径。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    #     - size：请求的建模样本规模；0 表示全量。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    # 输出参数：
    #     - (建模样本, 抽样报告)；说明或 NPZ 无效时返回 (None, None)。
    #-------------
    try:
        report = _read_cache_manifest(path.with_suffix('.json'))
        if (report is None or report['source']['sha256'] != source['sha256']
                or report['sample_size'] != size or report['seed'] != seed
                or not {'source_n', 'sample_n', 'group_summary', 'n_control', 'n_treatment'}.issubset(report)):
            return None, None
        sample = load_sample(path)
        expected = report['source_n'] if size == 0 else min(size, report['source_n'])
        if len(sample) != expected or report['sample_n'] != len(sample) or sample.row_id.max() >= report['source_n']:
            return None, None
        if (not all(sample[f].dtype == np.float32 for f in FEATURES)
                or not all(sample[label].dtype == np.int8 for label in LABELS)
                or sample.row_id.dtype != np.int64
                or int(sample.treatment.sum()) != report['n_treatment']
                or int((sample.treatment == 0).sum()) != report['n_control']):
            return None, None
        return sample, report
    except (KeyError, TypeError, ValueError, OSError, EOFError, BadZipFile, ZlibError):
        return None, None


def _audit_cache_result(path, source):
    #-------------
    # 校验全量审计缓存的来源和必需统计字段，返回可复用审计。
    # 输入参数：
    #     - path：全量审计缓存 JSON 的路径。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    # 输出参数：
    #     - 可复用的全量审计字典；来源或内容校验失败时返回 None。
    #-------------
    saved = _read_cache_manifest(path)
    if (saved is None or not isinstance(saved.get('source'), dict)
            or saved['source'].get('sha256') != source['sha256']):
        return None
    audit = saved.get('audit')
    fields = {
        'n_rows', 'n_treatment', 'n_control', 'feature_stats', 'groups', 'group_summary',
        'duplicate_row_fingerprints', 'duplicate_row_fraction', 'duplicate_fraction_denominator',
        'duplicate_count_definition', 'duplicate_method',
    }
    if not isinstance(audit, dict) or not fields.issubset(audit):
        return None
    return audit


def _load_diagnostics_cache(path, source, size, seed):
    #-------------
    # 校验诊断缓存并恢复 float64 数组；保留无限 SMD，不解析原始 CSV。
    # 输入参数：
    #     - path：诊断 NPZ 路径；说明文件使用相同文件名的 .json 后缀。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    #     - size：C2ST 两臂合计目标样本行数。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    # 输出参数：
    #     - 包含平衡表、C2ST 样本、SRM 和元数据的字典；未命中时返回 None。
    #-------------
    saved = _read_cache_manifest(path.with_suffix('.json'))
    if saved is None or not path.exists():
        return None
    if (not isinstance(saved.get('source'), dict)
            or saved['source'].get('sha256') != source['sha256']
            or saved.get('seed') != seed or saved.get('sample_size') != size):
        return None
    try:
        metadata, srm = saved['metadata'], saved['srm']
        with np.load(path, allow_pickle=False) as arrays:
            x, ids, labels = arrays['X'], arrays['row_id'], arrays['labels']
            if x.dtype != np.float64 or x.shape != (size, len(FEATURES)):
                return None
            check_feature_matrix(x, n_features=len(FEATURES))
            check_row_ids(ids, n_rows=size, require_integer=True)
            if (labels.shape != (size, len(LABELS)) or labels.dtype != np.int8
                    or ids.dtype != np.int64 or np.any(ids >= metadata['n'])
                    or np.any(np.diff(ids) <= 0)):
                return None
            for j, label in enumerate(LABELS):
                check_binary_vector(labels[:, j], label, size)
            if int(labels[:, 0].sum()) != size // 2:
                return None
            sample = pd.DataFrame(x, columns=FEATURES)
            for j, label in enumerate(LABELS):
                sample[label] = labels[:, j].astype(np.int8)
            sample['row_id'] = ids
            balance = pd.DataFrame({name: arrays['balance_' + name] for name in _BALANCE_COLUMNS})
        if (balance.feature.tolist() != FEATURES
                or not balance.source_sha256.eq(source['sha256']).all()
                or metadata['n'] != metadata['n_control'] + metadata['n_treatment']):
            return None
        for arm in ('control', 'treatment'):
            key = 'n_' + arm
            if not balance[key].eq(metadata[key]).all() or srm[key] != metadata[key]:
                return None
        return {'balance': balance, 'c2st_sample': sample, 'srm': srm, 'metadata': metadata}
    except (KeyError, TypeError, ValueError, OSError, EOFError, BadZipFile, ZlibError):
        return None


def _write_diagnostics_cache(result, path, source, size, seed):
    #-------------
    # 保存 C2ST 样本和全量统计，最后写入说明，避免复用不完整缓存。
    # 输入参数：
    #     - result：本次生成的诊断结果字典，包含平衡表、C2ST 样本、SRM 和元数据。
    #     - path：诊断 NPZ 的输出路径；同名 JSON 保存说明。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    #     - size：C2ST 两臂合计目标样本行数。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    # 输出参数：
    #     - 无返回值；写入诊断 NPZ，再写入 JSON 说明。
    #-------------
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample, balance = result['c2st_sample'], result['balance']
    arrays = {
        'X': sample[FEATURES].to_numpy(dtype=np.float64),
        'labels': sample[LABELS].to_numpy(dtype=np.int8),
        'row_id': sample.row_id.to_numpy(dtype=np.int64),
    }
    for name in _BALANCE_COLUMNS:
        values = balance[name].to_numpy()
        arrays['balance_' + name] = values.astype(str) if values.dtype.kind == 'O' else values
    np.savez_compressed(path, **arrays)
    write_result({
        'source': source,
        'seed': seed, 'sample_size': size,
        'metadata': result['metadata'], 'srm': result['srm'],
    }, path.with_suffix('.json'))


def prepare_sample(args, *, root: Path = ROOT, progress=None, chunk_size=250_000):
    #-------------
    # 功能：固定准备样本、审计和诊断；只扫描缓存缺失的任务，三项共用一次读取。
    # 输入参数：
    #     - args：运行参数对象，包含数据路径、种子及样本规模参数。
    #     - root：项目根目录，缓存位于其 artifacts/cache 下。
    #     - progress：进度回调；None 时使用立即刷新的标准输出。
    #     - chunk_size：每块行数，默认 250_000。
    # 输出参数：
    #     - (建模样本, 全量审计, 来源指纹, 诊断及抽样说明字典)。
    #-------------
    size, diagnostic_size = args.sample_size, args.c2st_sample
    if diagnostic_size < 20 or diagnostic_size % 20:
        raise ValueError('C2ST sample_size 必须大于等于 20 且能被 20 整除')
    per_arm = diagnostic_size // 2
    progress = progress if progress is not None else lambda message: print(message, flush=True)

    current = fingerprint(args.data)
    cache_dir = safe_output(Path(root) / 'artifacts/cache')
    sample_path = cache_dir / f'sample_{size}_seed{args.seed}.npz'
    audit_path = cache_dir / f'full_audit_{current["sha256"]}.json'
    diagnostic_path = cache_dir / f'c2st_sample_n{diagnostic_size}_seed{args.seed}.npz'
    sample, report = _load_sample_cache(sample_path, current, size, args.seed)
    audit = _audit_cache_result(audit_path, current)
    diagnostics = _load_diagnostics_cache(diagnostic_path, current, diagnostic_size, args.seed)


    needs_sample = report is None
    needs_audit = audit is None
    needs_diagnostics = diagnostics is None
    progress('Preparation: ' + '; '.join(
        f'{name}={"scan" if needed else "cache"}'
        for name, needed in zip(('Sample', 'Audit', 'Diagnostics'),
                                (needs_sample, needs_audit, needs_diagnostics))
    ))
    scanned = None
    if any((needs_sample, needs_audit, needs_diagnostics)):
        scanned = scan_raw_data(
            args.data, sample_size=size, sample_enabled=needs_sample, seed=args.seed,
            chunk_size=chunk_size, per_arm=per_arm if needs_diagnostics else None,
            audit_enabled=needs_audit, diagnostics_enabled=needs_diagnostics,
            progress=progress, source=current,
        )
        if needs_sample:
            sample, report = scanned['sample'], scanned['sample_report']
        if needs_audit:
            audit = scanned['audit']
        if needs_diagnostics:
            diagnostics = {key: scanned[key] for key in ('balance', 'c2st_sample', 'srm', 'metadata')}


    if needs_sample:
        save_sample(sample, sample_path)
        write_result(report, sample_path.with_suffix('.json'))
    if needs_audit:
        write_result({'source': current, 'audit': audit}, audit_path)
    if needs_diagnostics:
        _write_diagnostics_cache(diagnostics, diagnostic_path, current, diagnostic_size, args.seed)

    prepared = {key: diagnostics[key] for key in ('balance', 'c2st_sample', 'srm')}
    prepared['metadata'] = scan_metadata(
        total=audit['n_rows'], counts=[audit['n_control'], audit['n_treatment']],
        source=current, seed=args.seed, chunk_size=chunk_size, per_arm=per_arm,
        audit_enabled=True, diagnostics_enabled=True, sample_enabled=True,
        c2st_sample=prepared['c2st_sample'],
    )
    prepared['metadata']['task_status'] = {
        name: 'computed' if needed else 'cache'
        for name, needed in zip(('sample', 'audit', 'diagnostics'),
                                (needs_sample, needs_audit, needs_diagnostics))
    }
    prepared.update(sample_report=report, source=current)
    return sample, audit, current, prepared


def split_sample(df, outcome='conversion', seed=42):
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
