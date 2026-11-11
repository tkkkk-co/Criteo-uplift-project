#-------------
# 统一管理数据准备任务的开关、独立缓存以及建模样本划分。
#-------------
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, FEATURES, LABELS
from .input_checks import check_binary_vector, check_feature_matrix, check_row_ids
from .pipeline_io import fingerprint, load_sample, safe_output, save_sample, write_result
from .raw_scan import SAMPLE_CACHE_VERSION, scan_raw_data, scan_metadata, validate_scan_options

_BALANCE_COLUMNS = [
    'feature', 'n_treatment', 'n_control', 'mean_treatment', 'mean_control',
    'smd', 'abs_smd', 'ks_statistic', 'ks_pvalue', 'scope', 'representation', 'source_sha256',
]


def _read_cache_manifest(path):
    #-------------
    # 读取缓存说明；文件缺失、内容截断或 JSON 无效时视为缓存未命中。
    #-------------
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sample_cache_report(path, source, size, seed):
    #-------------
    # 校验建模样本缓存的来源、规模、种子、表示和报告完整性。
    #-------------
    info = _read_cache_manifest(path.with_suffix('.json'))
    fields = {
        'scope', 'sample_cache_version', 'source', 'source_n', 'sample_size', 'sample_n',
        'seed', 'sampling', 'feature_representation', 'n_treatment', 'n_control', 'group_summary',
    }
    if not path.exists() or info is None or not fields.issubset(info):
        return None
    if (not isinstance(info['source'], dict)
            or not isinstance(info['group_summary'], list)
            or not all(isinstance(row, dict) for row in info['group_summary'])):
        return None
    if (info['scope'] != 'modeling_sample'
            or info['source'].get('sha256') != source['sha256']
            or info['sample_size'] != size or info['seed'] != seed
            or info['sample_cache_version'] != SAMPLE_CACHE_VERSION
            or info['feature_representation'] != 'float32'
            or [row.get('group') for row in info['group_summary']] != ['Control', 'Treatment', 'Total']):
        return None
    return info


def _audit_cache_result(path, source):
    #-------------
    # 校验全量审计缓存的来源和必需统计字段，返回可复用审计。
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
    #-------------
    saved = _read_cache_manifest(path.with_suffix('.json'))
    if saved is None or not path.exists():
        return None
    if (not isinstance(saved.get('source'), dict)
            or saved['source'].get('sha256') != source['sha256']
            or saved.get('seed') != seed or saved.get('sample_size') != size
            or saved.get('payload_sha256') != fingerprint(path)['sha256']):
        return None
    try:
        metadata, srm = saved['metadata'], saved['srm']
        if (metadata['source']['sha256'] != source['sha256']
                or metadata['seed'] != seed or metadata['sample_n'] != size
                or metadata['features'] != FEATURES
                or metadata['validated_columns'] != FEATURES + LABELS
                or metadata['representation'] != 'raw_csv_float64'
                or metadata['diagnostics_enabled'] is not True
                or metadata['c2st_sample_enabled'] is not True
                or metadata['rows_dropped'] != 0
                or metadata['sample_n_control'] != size // 2
                or metadata['sample_n_treatment'] != size // 2):
            return None
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
    except (KeyError, TypeError, ValueError, OSError, EOFError, BadZipFile):
        return None


def _write_diagnostics_cache(result, path, source, size, seed):
    #-------------
    # 保存 C2ST 样本和全量统计，最后写入说明及载荷指纹，避免复用不完整缓存。
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
        'seed': seed, 'sample_size': size, 'payload_sha256': fingerprint(path)['sha256'],
        'metadata': result['metadata'], 'srm': result['srm'],
    }, path.with_suffix('.json'))


def _validate_prepared_counts(sample, report, audit, diagnostics, scanned, size):
    #-------------
    # 核对各任务缓存中的全量人数及抽样行数；不要求必须启用审计。
    #-------------
    populations = []
    if report is not None:
        populations.append(report['source_n'])
        expected = report['source_n'] if size == 0 else min(size, report['source_n'])
        if len(sample) != expected or report['sample_n'] != len(sample):
            raise ValueError('抽样缓存行数与配置及原始数据规模不一致')
        if np.any(sample.row_id.to_numpy() >= report['source_n']):
            raise ValueError('抽样缓存 row_id 超出原始数据规模')
    if audit is not None:
        populations.append(audit['n_rows'])
    for result in (diagnostics, scanned):
        if result is not None:
            metadata = result['metadata']
            populations.append(metadata['n'])
            if audit is not None:
                for key in ('n_treatment', 'n_control'):
                    if audit[key] != metadata[key]:
                        raise ValueError(f'全量审计缓存与当前扫描不一致：{key}')
    if len(set(populations)) != 1:
        raise ValueError('缓存与当前原始数据规模不一致')
    return populations[0]


def prepare_sample(args, *, root: Path = ROOT, progress=None, sample_enabled=None,
                   audit_enabled=None, diagnostics_enabled=None, chunk_size=None):
    #-------------
    # 功能：决定需要重算的任务，共用一次扫描，独立读写各自缓存。
    # 参数：三个开关优先采用显式参数，否则读取 args 的同名属性，未提供时默认开启。
    #       关闭的任务不读取或写入缓存，也不要求提供该任务的专用参数。
    #       chunk_size 默认读取 args.chunk_size，未提供时为 250_000。
    # 返回：(建模样本、全量审计、来源指纹、诊断及抽样说明字典)。
    #       未启用任务返回 None；第四项不重复持有建模大表或审计字典。
    #       C2ST 拟合仍由 run_diagnostics() 执行，原 CLI 调用方式不变。
    #-------------
    switches = [sample_enabled, audit_enabled, diagnostics_enabled]
    for i, name in enumerate(('sample_enabled', 'audit_enabled', 'diagnostics_enabled')):
        switches[i] = getattr(args, name, True) if switches[i] is None else switches[i]
        if not isinstance(switches[i], bool):
            raise ValueError(f'{name} 必须为布尔值')
    sample_enabled, audit_enabled, diagnostics_enabled = switches
    if not any(switches):
        raise ValueError('至少启用一个数据准备任务')
    chunk_size = getattr(args, 'chunk_size', 250_000) if chunk_size is None else chunk_size
    size = args.sample_size if sample_enabled else None
    diagnostic_size = args.c2st_sample if diagnostics_enabled else None
    if sample_enabled and size is None:
        raise ValueError('sample_size 必须为大于等于 0 的整数')
    if diagnostics_enabled:
        if (isinstance(diagnostic_size, bool) or not isinstance(diagnostic_size, int)
                or diagnostic_size < 20 or diagnostic_size % 20):
            raise ValueError('C2ST sample_size 必须大于等于 20 且能被 20 整除')
        if args.permutations < 1:
            raise ValueError('permutations 必须为正数')
    per_arm = diagnostic_size // 2 if diagnostics_enabled else None
    validate_scan_options(size, chunk_size, per_arm, audit_enabled, diagnostics_enabled)
    progress = progress if progress is not None else lambda message: print(message, flush=True)
    current = fingerprint(args.data)
    cache_dir = safe_output(Path(root) / 'artifacts/cache')
    sample_path = cache_dir / f'sample_{size}_seed{args.seed}.npz'
    audit_path = cache_dir / f'full_audit_{current["sha256"]}.json'
    diagnostic_path = cache_dir / f'full_diagnostics_{current["sha256"]}_n{diagnostic_size}_seed{args.seed}.npz'

    report = _sample_cache_report(sample_path, current, size, args.seed) if sample_enabled else None
    sample = None
    audit = _audit_cache_result(audit_path, current) if audit_enabled else None
    diagnostics = _load_diagnostics_cache(diagnostic_path, current, diagnostic_size, args.seed) if diagnostics_enabled else None
    needs_sample = sample_enabled and report is None
    needs_audit = audit_enabled and audit is None
    needs_diagnostics = diagnostics_enabled and diagnostics is None
    progress('Preparation: ' + '; '.join(
        f'{name}={"disabled" if not enabled else "scan" if needed else "cache"}'
        for name, enabled, needed in zip(
            ('Sample', 'Audit', 'Diagnostics'), switches,
            (needs_sample, needs_audit, needs_diagnostics),
        )
    ))
    scanned = None
    if any((needs_sample, needs_audit, needs_diagnostics)):
        scanned = scan_raw_data(
            args.data, sample_size=size if needs_sample else None, seed=args.seed,
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
    # 缓存命中的建模大表延后加载，避免它与审计指纹或全量诊断临时数组同时占用内存。
    if sample_enabled and not needs_sample:
        sample = load_sample(sample_path)
    total = _validate_prepared_counts(sample, report, audit, diagnostics, scanned, size)
    if needs_sample:
        save_sample(sample, sample_path)
        write_result(report, sample_path.with_suffix('.json'))
    if needs_audit:
        write_result({'source': current, 'audit': audit}, audit_path)
    if needs_diagnostics:
        _write_diagnostics_cache(diagnostics, diagnostic_path, current, diagnostic_size, args.seed)
    if diagnostics is not None:
        prepared = dict(diagnostics)
    elif scanned is not None:
        prepared = {key: scanned[key] for key in ('balance', 'c2st_sample', 'srm', 'metadata')}
    else:
        # 根据已启用任务的缓存恢复来源说明；建模报告没有全量两臂人数，未知计数保留为空。
        counts = [audit['n_control'], audit['n_treatment']] if audit is not None else [None, None]
        prepared = {'balance': None, 'c2st_sample': None, 'srm': None, 'metadata': scan_metadata(
            total=total, counts=counts, source=current, seed=args.seed, chunk_size=chunk_size,
            per_arm=None, audit_enabled=False, diagnostics_enabled=False,
            model_sample_enabled=False, c2st_sample_enabled=False, c2st_sample=None,
        )}
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
