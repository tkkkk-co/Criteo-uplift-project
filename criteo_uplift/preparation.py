"""一次流式扫描完成入口检查、全量审计、抽样；复用独立缓存后划分样本。"""
import json
from contextlib import ExitStack
from tempfile import TemporaryFile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import ROOT, DEFAULT_DATA, FEATURES, LABELS, COLUMNS
from .input_checks import check_binary_vector, check_feature_matrix, check_row_ids
from .pipeline_io import fingerprint, load_sample, safe_output, save_sample, write_result
from .diagnostics import covariate_balance, srm_summary

SAMPLE_CACHE_VERSION = 1
AUDIT_CACHE_VERSION = 1

def _scan_raw_data(
    path=DEFAULT_DATA, sample_size=1_000_000, seed=42, chunk_size=250_000,
    *, per_arm=None, audit_enabled=True, diagnostics_enabled=True,
    progress=print, source=None,
):
    # 1. 检查参数、文件来源和列名，确定本次需要执行哪些任务。
    path = Path(path)
    if sample_size is not None and (
        isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0
    ):
        raise ValueError('sample_size must be an integer >= 0 or None')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if per_arm is not None and (
        isinstance(per_arm, bool) or not isinstance(per_arm, int) or per_arm < 2
    ):
        raise ValueError('per_arm must be an integer >= 2')
    if not isinstance(audit_enabled, bool) or not isinstance(diagnostics_enabled, bool):
        raise ValueError('audit_enabled and diagnostics_enabled must be booleans')

    model_sample_enabled = sample_size is not None
    c2st_sample_enabled = per_arm is not None

    if not any((audit_enabled, diagnostics_enabled, model_sample_enabled, c2st_sample_enabled)):
        raise ValueError('Enable at least one scan task')
    source = fingerprint(path) if source is None else source
    if list(pd.read_csv(path, nrows=0).columns) != COLUMNS:
        raise ValueError(f'CSV schema must be exactly {COLUMNS}')

    # 2. 准备两个用途不同的抽样任务。保留原种子规则，诊断抽样不消耗建模抽样的随机数。
    # 建模使用一个全局样本池；C2ST 使用 Control、Treatment 各一个样本池。
    rng = np.random.default_rng(seed)
    reservoir = np.empty((0, len(COLUMNS)), dtype=np.float64)
    row_ids = np.empty(0, dtype=np.int64)
    priorities = np.empty(0)
    generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    diagnostic_reservoirs = [pd.DataFrame(), pd.DataFrame()]
    diagnostic_keys = [np.empty(0), np.empty(0)]

    # 3. 准备审计累计量。人数由所有任务共用，重复行仅统计、不删除。
    counts = np.zeros(2, dtype=np.int64)            # 累计全量两组人数：索引 0 为 Control，索引 1 为 Treatment。
    positives = np.zeros((2, 3), dtype=np.int64)    # 累计全量标签正例数：两行对应 Control、Treatment，三列对应 conversion、visit、exposure。
    hashes, all_chunks = [], []                     # hashes 保存全量各行的内容指纹，用于统计重复行；all_chunks 仅在 sample_size=0 时保存全部建模数据块。

    # 以下四项仅在开启审计、关闭完整诊断时累计；开启完整诊断时复用后面的逐列检查结果。
    means = np.zeros((2, len(FEATURES)))             # 保存全量两组各特征的累计均值。
    centered_squares = np.zeros_like(means)          # 保存各组相对均值的偏差平方和，避免近常量特征计算方差时相减失去精度。
    minimum = np.full(len(FEATURES), np.inf)        # 记录全量各特征的最小值，不分组；初始为正无穷，后续由实际值更新。
    maximum = np.full(len(FEATURES), -np.inf)       # 记录全量各特征的最大值，不分组；初始为负无穷，后续由实际值更新。
    total = 0                                       # 累计已读取的原始数据总行数，同时确定下一块数据的原始行号起点。
    bad_visit = 0                                   # 累计全量数据中 conversion=1 但 visit=0 的行数，用于检查标签逻辑异常。
    control_exposure = 0                            # 累计全量 Control 组中 exposure=1 的行数，用于检查对照组曝光情况。
    feature_stats, records = [], []                 # feature_stats 保存各特征的全量审计统计；records 保存各特征的全量平衡检查结果，包含 SMD、KS。

    #-----------------------------扫描开始----------------------------
    # 4. 按需创建临时文件。diagnostics_enabled打开的时候，全量 KS 需要每个特征的全部取值，因此逐列把feature和treatment写入临时二进制文件。
    # ExitStack 在结束或报错时关闭 CSV 读取器和所有临时文件。
    with ExitStack() as stack:
        files = {
            name: stack.enter_context(TemporaryFile())
            for name in FEATURES + ['treatment']
        } if diagnostics_enabled else {}
        #CSV 分块读取器
        reader = stack.enter_context(pd.read_csv(path, dtype=np.float64, chunksize=chunk_size))

        for chunk in reader:
            # 5.1 共用的读取与校验：检查当前块的全部列，并分配原始行号。
            values = chunk.to_numpy()
            check_feature_matrix(values)
            for label in LABELS:
                check_binary_vector(chunk[label], label, len(chunk))
            ids = np.arange(total, total + len(chunk), dtype=np.int64)
            check_row_ids(ids, n_rows=len(chunk), require_integer=True)
            treatment = chunk.treatment.to_numpy(dtype=np.int8)
            x = values[:, :len(FEATURES)]
            for arm in (0, 1):
                #更新全量分组计数：
                counts[arm] += int((treatment == arm).sum())

            # 5.2 全量审计：累计重复行指纹、结果正例数和标签逻辑异常人数。
            if audit_enabled:
                hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
                for arm in (0, 1):
                    mask = treatment == arm #布尔数组，标记当前块中哪些行属于这一组
                    positives[arm] += values[mask, len(FEATURES) + 1:].sum(axis=0).astype(np.int64)
                    # 未启用完整平衡检查时，需要逐块累计特征统计；启用时程序会把全量特征写入临时文件 之后利用covariate_balance 计算均值 SMD KS
                    if not diagnostics_enabled:
                        sub = x[mask]
                        if len(sub):
                            # 合并旧数据，分别在T/C组下计算均值 方差 用于后续计算SMD
                            old_n = counts[arm] - len(sub)
                            chunk_mean = sub.mean(axis=0)
                            difference = chunk_mean - means[arm] #新旧两批数据的均值差，means[arm] 仍然是合并当前数据块之前的均值
                            #更新分组方差：分批合并方差统计量经典公式/算法，能够避免保存全部历史数据
                            centered_squares[arm] += (
                                np.square(sub - chunk_mean).sum(axis=0)
                                + np.square(difference) * old_n * (len(sub) / counts[arm])
                            )
                            #更新分组均值：
                            means[arm] += difference * (len(sub) / counts[arm])
                if not diagnostics_enabled:
                    minimum = np.minimum(minimum, x.min(axis=0))
                    maximum = np.maximum(maximum, x.max(axis=0))
                bad_visit += int(((chunk.conversion == 1) & (chunk.visit == 0)).sum())
                control_exposure += int(((chunk.treatment == 0) & (chunk.exposure == 1)).sum())

            # 5.3 建模抽样：为全体行分配随机优先级，不按 Treatment 或结果标签设置配额。
            # sample_size=0 时保留所有行，否则只保留指定数量的候选样本。
            if model_sample_enabled:
                if sample_size == 0:
                    model_chunk = chunk.astype({
                        **{f: 'float32' for f in FEATURES},
                        **{label: 'int8' for label in LABELS},
                    })
                    model_chunk.insert(0, 'row_id', ids)
                    all_chunks.append(model_chunk)
                else:
                    reservoir = np.concatenate([reservoir, values])
                    priorities = np.concatenate([priorities, rng.random(len(chunk))])
                    row_ids = np.concatenate([row_ids, ids])
                    if len(reservoir) > sample_size:
                        keep = np.argpartition(priorities, sample_size - 1)[:sample_size]
                        reservoir, priorities, row_ids = reservoir[keep], priorities[keep], row_ids[keep]

            if diagnostics_enabled:
                # 5.4 全量平衡检查的数据准备：将当前块的特征和分组标签追加到对应文件。
                for feature in FEATURES:
                    chunk[feature].to_numpy().tofile(files[feature])
                treatment.tofile(files['treatment'])

            # 5.5 C2ST 抽样：单独受 per_arm 控制，不依赖全量诊断开关。
            # 两组分别等概率、无放回抽取 per_arm 行，保留 float64 特征。
            if c2st_sample_enabled:
                diagnostic_chunk = chunk.assign(row_id=ids)
                for arm in (0, 1):
                    arm_rows = diagnostic_chunk.loc[treatment == arm]
                    combined_keys = np.concatenate([
                        diagnostic_keys[arm], generators[arm].random(len(arm_rows)),
                    ])
                    combined = pd.concat([diagnostic_reservoirs[arm], arm_rows], ignore_index=True)
                    if len(combined) > per_arm:
                        keep = np.argpartition(combined_keys, per_arm - 1)[:per_arm]
                        combined, combined_keys = combined.iloc[keep], combined_keys[keep]
                    diagnostic_reservoirs[arm], diagnostic_keys[arm] = combined, combined_keys

            total += len(chunk)
            if total % 1_000_000 == 0:
                progress(f'Raw scan: {total:,} rows')

        # 6. 扫描结束：确认两组都有数据，C2ST 抽样人数足够；逐列完成全量 SMD/KS。
        if total == 0 or (counts == 0).any():
            raise ValueError('数据为空或缺少 treatment/control')
        if c2st_sample_enabled and min(counts) < per_arm:
            raise ValueError(f'Need {per_arm} rows per arm; found {counts.tolist()}')
        progress(f'Raw scan: n={total:,}; control={counts[0]:,}; treatment={counts[1]:,}')
        if diagnostics_enabled:
            for file in files.values():
                file.flush()
            treatment = np.memmap(files['treatment'], dtype=np.int8, mode='r', shape=(total,))
            for feature in FEATURES:
                values = np.memmap(files[feature], dtype=np.float64, mode='r', shape=(total,))
                frame = pd.DataFrame({feature: values, 'treatment': treatment}, copy=False)
                record = covariate_balance(frame, [feature]).iloc[0].to_dict()
                records.append(record)
                if audit_enabled:
                    # 审计复用covariant-balance输出的record结果，再补齐范围与组内标准差。
                    row = {'feature': feature, 'min': float(values.min()), 'max': float(values.max())}
                    for arm, name in ((0, 'control'), (1, 'treatment')):
                        row[f'mean_{arm}'] = record[f'mean_{name}']
                        row[f'std_{arm}'] = float(values[treatment == arm].std(ddof=1))
                    row['smd'] = record['smd'] if np.isfinite(record['smd']) else np.nan
                    row['smd_status'] = 'computed' if np.isfinite(row['smd']) else 'undefined_zero_or_missing_variance'
                    feature_stats.append(row)
                progress(f"Full {feature}: SMD={record['smd']:.6g}; KS={record['ks_statistic']:.6g}")
                del frame, values
            del treatment
    #-----------------------------扫描结束----------------------------

    # 7. 整理建模样本、C2ST抽样样本、全量审计、协变量诊断
    sample, sample_report, audit = None, None, None
    balance, c2st_sample, srm = None, None, None

    if model_sample_enabled:
        if sample_size == 0:
            sample = pd.concat(all_chunks, ignore_index=True)
        else:
            order = np.argsort(row_ids)
            sample = pd.DataFrame(reservoir[order], columns=COLUMNS).astype({
                **{f: 'float32' for f in FEATURES}, **{label: 'int8' for label in LABELS},
            })
            sample.insert(0, 'row_id', row_ids[order])
        # 抽样完成时生成说明；来源总行数直接使用本次扫描计数，不依赖审计开关。
        sample_report = {
            'scope': 'modeling_sample',
            'sample_cache_version': SAMPLE_CACHE_VERSION,
            'source': source,
            'source_n': total,
            'sample_size': sample_size,
            'sample_n': len(sample),
            'seed': seed,
            'sampling': (
                'all rows' if sample_size == 0
                else 'uniform without replacement, random-priority reservoir'
            ),
            'feature_representation': 'float32',
            'n_treatment': int(sample.treatment.sum()),
            'n_control': int((sample.treatment == 0).sum()),
            # 抽样报告自带可展示的分组汇总，Notebook 无需重读样本或汇总划分表。
            'group_summary': [
                {
                    'group': name,
                    'n': len(rows),
                    'fraction': len(rows) / len(sample),
                    'conversion_positive': int(rows.conversion.sum()),
                    'conversion_rate': float(rows.conversion.mean()),
                }
                for name, rows in (
                    ('Control', sample[sample.treatment == 0]),
                    ('Treatment', sample[sample.treatment == 1]),
                    ('Total', sample),
                )
            ],
        }

    if c2st_sample_enabled:
        c2st_sample = pd.concat(diagnostic_reservoirs, ignore_index=True).sort_values('row_id').reset_index(drop=True)
        for label in LABELS:
            c2st_sample[label] = c2st_sample[label].astype(np.int8)
        c2st_sample['row_id'] = c2st_sample.row_id.astype(np.int64)
        if not c2st_sample.row_id.is_unique or len(c2st_sample) != 2 * per_arm:
            raise AssertionError('Invalid diagnostic reservoir')

    if audit_enabled:
        if not diagnostics_enabled:
            # 未开启全量诊断：用扫描时累计的统计量，整理每个特征的审计记录。
            for j, feature in enumerate(FEATURES):
                row = {'feature': feature, 'min': minimum[j], 'max': maximum[j]}
                for arm in (0, 1):
                    row[f'mean_{arm}'] = means[arm, j]
                    row[f'std_{arm}'] = (
                        # 偏差平方和除以自由度，再开方，得到样本标准差。
                        np.sqrt(max(0, centered_squares[arm, j] / (counts[arm] - 1)))
                        if counts[arm] > 1 else np.nan
                    )
                denominator = np.sqrt((row['std_0'] ** 2 + row['std_1'] ** 2) / 2)
                difference = row['mean_1'] - row['mean_0']
                row['smd'] = difference / denominator if denominator > 0 else (
                    0.0 if denominator == 0 and difference == 0 else np.nan
                )
                row['smd_status'] = 'computed' if np.isfinite(row['smd']) else 'undefined_zero_or_missing_variance'
                feature_stats.append(row)
        # 已开启全量诊断时，feature_stats 已在上面的逐列诊断中生成，直接复用。
        # 多余行数 = 总行数 - 不同指纹数；同一内容出现 k 次贡献 k-1 行。
        duplicate_excess = total - int(np.unique(np.concatenate(hashes)).size)
        #输出完整审计字典
        audit = {
            'n_rows': total, 'n_columns': len(COLUMNS),
            'n_treatment': int(counts[1]), 'n_control': int(counts[0]),
            'treatment_ratio': counts[1] / total,
            'missing_or_nonfinite': 0, 'invalid_binary_labels': 0,
            'control_exposed': control_exposure, 'conversion_without_visit': bad_visit,
            #原始 CSV 的所有全量样本进行重复行检查
            'duplicate_row_fingerprints': duplicate_excess,
            'duplicate_row_fraction': duplicate_excess / total,
            'duplicate_fraction_denominator': total,
            'duplicate_count_definition': (
                '多余行数 = 全量行数 - 不同完整行指纹数；同一指纹出现 k 次贡献 k-1 行。'
                '不是出现重复的不同记录种数，也不是所有参与重复的行数；比例分母为全量原始记录数。'
            ),
            'duplicate_method': (
                '对原始 CSV 的全部 16 个字段按 float64 表示，使用 pandas.util.hash_pandas_object'
                '(index=False) 生成 uint64 指纹，并跨全部分块统计不同指纹数。'
                '64 位哈希存在碰撞可能，float64 解析也不等于原始文本逐字节相等；仅识别内容重复候选。'
                '没有可靠用户标识，不能证明同一用户重复出现，不据此自动删除记录。'
            ),
            'feature_stats': feature_stats,
            'groups': {
                str(arm): {
                    'n': int(counts[arm]),
                    **{label: {'positive': positives[arm, k], 'rate': positives[arm, k] / counts[arm]}
                       for k, label in enumerate(LABELS[1:])},
                } for arm in (0, 1)
            },
            # 与 sample_report 使用同一份展示字段；总体行也在计算阶段准备好。
            'group_summary': [
                {
                    'group': name,
                    'n': int(n),
                    'fraction': float(n / total),
                    'conversion_positive': int(positive),
                    'conversion_rate': float(positive / n),
                }
                for name, n, positive in (
                    ('Control', counts[0], positives[0, 0]),
                    ('Treatment', counts[1], positives[1, 0]),
                    ('Total', total, positives[:, 0].sum()),
                )
            ],
        }

    if diagnostics_enabled:
        balance = pd.DataFrame(records).assign(
            scope='full_published_dataset', representation='raw_csv_float64', source_sha256=source['sha256'],
        )
        srm = {
            **srm_summary(int(counts[1]), int(counts[0]), expected_ratio=.85),
            'scope': 'full_published_dataset', 'source_sha256': source['sha256'],
            'reference_source': 'https://arxiv.org/html/2111.10106#S3',
            'interpretation': '公开数据分配计数与85%外部参考比例的一致性；不验证原始线上实验SRM或随机化。',
        }

    # 来源和全量人数始终记录；未执行的抽样或检验不填写对应方法。
    metadata = {
        'scope': 'full_published_dataset', 'source': source, 'representation': 'raw_csv_float64',
        'features': FEATURES, 'validated_columns': COLUMNS,
        'n': total, 'n_treatment': int(counts[1]), 'n_control': int(counts[0]),
        'rows_dropped': 0, 'seed': seed, 'chunk_size': chunk_size,
        'audit_enabled': audit_enabled, 'diagnostics_enabled': diagnostics_enabled,
        'model_sample_enabled': model_sample_enabled, 'c2st_sample_enabled': c2st_sample_enabled,
        'sample_n': len(c2st_sample) if c2st_sample_enabled else None,
        'sample_n_treatment': per_arm, 'sample_n_control': per_arm,
        'sampling': 'independent uniform-priority reservoir without replacement within each arm' if c2st_sample_enabled else None,
        'sample_scope': 'full_source_arm_balanced_sample' if c2st_sample_enabled else None,
        'sampling_target': 'P(X|T=1) versus P(X|T=0); balanced arms, not population T prevalence' if c2st_sample_enabled else None,
        'row_id_definition': 'zero-based original CSV data-row position; not a user ID',
        'ks_method': 'two-sided asymptotic; unadjusted exploratory p-values; ties and dependence limit calibration' if diagnostics_enabled else None,
    }

    # 8. 核验扫描期间原始文件未改变，统一返回结果。保存 CSV/JSON 和运行 C2ST 交给上层函数。
    if fingerprint(path) != source:
        raise RuntimeError('Raw source changed during scan')
    return {
        'sample': sample, 'sample_report': sample_report, 'audit': audit, 'balance': balance,
        'c2st_sample': c2st_sample, 'srm': srm, 'metadata': metadata, 'source': source,
    }

def prepare_sample(args, *, root: Path = ROOT, progress=None):
    """核验并复用缓存，返回建模样本、全量审计、文件指纹和扫描结果。"""
    # 扫描和缓存模块引用这里的基础定义，在准备样本时再导入。

    if progress is None:
        progress = lambda message: print(message, flush=True)

    if args.c2st_sample < 20 or args.c2st_sample % 20:
        raise ValueError('C2ST sample_size must be >=20 and divisible by 20')
    if args.permutations < 1:
        raise ValueError('permutations must be positive')

    # 1. 分开定位抽样结果、抽样说明和按原始内容指纹共享的全量审计。
    base = safe_output(Path(root) / 'artifacts')
    base.mkdir(exist_ok=True)
    cache = base / f'cache/sample_{args.sample_size}_seed{args.seed}.npz' #搜索对应args的抽样样本缓存
    metadata = cache.with_suffix('.json') # 上述抽样样本的说明信息
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

    # 4. 调用_scan_raw_data() 完成全量审计、抽样（建模抽样、C2ST抽样）、诊断（SRM SMD KS）
    progress(
        '统一扫描：全量检查 + C2ST 抽样；'
        + ('复用建模缓存' if reuse_sample else '重新建模抽样') + '；'
        + ('复用全量审计' if reuse_audit else '重新全量审计')
    )
    scanned = _scan_raw_data(
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



#-------------------------------------------------------------------------

def split_sample(df, outcome='conversion', seed=42):
    """按处理组和结果分层，返回三个数据集、人数汇总和重叠检查。"""
    if outcome not in ['conversion', 'visit']:
        raise ValueError('unknown outcome')
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
            raise ValueError(f'{name} split size differs from the 60/20/20 design')
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
