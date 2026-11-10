#-----------------------------------------------------------------------------------------------------------------------
# 统一定义全量读取、审计、建模与 C2ST 抽样，以及数据诊断函数。
# Notebook 01–02 的主要计算模块；调用链和输出文件见 docs/RESULTS_GUIDE.md。
# scan_raw_data 负责扫描/统计/抽样，update_experiment_diagnostics 负责 C2ST 与诊断产物落盘。
# 阅读顺序：scan_raw_data → 审计与平衡检查、抽样（项目抽样和C2ST抽样）→ C2ST → 结果保存。
# scan_raw_data 是唯一的全量扫描入口，由 run_pipeline.py 等调用。
#
# 导入本模块时只加载依赖、初始化常量并定义函数，不自动运行诊断。
# 本文件没有 __main__ 执行入口；函数被调用时才读取数据、训练或检验，
# 其中 update_experiment_diagnostics 还负责保存诊断结果。
#
# 这些检查只使用处理前特征 f0–f11。
# 预测实验分组只能诊断可观测的不平衡；它不能证明随机化成立，也不能排除隐藏偏差。
# 结果可学习性并不是对处理效应异质性的检验。
#-----------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from tempfile import TemporaryFile
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .preparation import COLUMNS, FEATURES, LABELS, DEFAULT_DATA, fingerprint
from .pipeline_io import safe_output, write_result

_PRETREATMENT_FEATURES = frozenset(f"f{i}" for i in range(12))
SAMPLE_CACHE_VERSION = 1


#-----------------------------------------------------------------------------------------------------------------------
# 函数：scan_raw_data —— 一次分块扫描，统一准备审计、SRM、协变量平衡检验（SMD KS）和抽样（项目抽样、C2ST抽样）
# 输入：
#     path：原始 CSV 文件路径。
#     sample_size：建模样本行数上限；0 表示保留全量，None 只表示跳过建模抽样。项目默认100万。
#     seed：随机种子，默认 42。
#     chunk_size：每块读取的行数，默认 250,000。
#     per_arm：C2ST 每组样本数；None 只表示跳过 C2ST 抽样。项目主流程默认每组1万。
#     audit_enabled：是否执行全量审计，默认 True，与是否抽样无关。
#     diagnostics_enabled：是否执行全量 SRM、SMD、KS，默认 True，与是否抽样无关。
#     progress：接收进度文字的函数，默认 print。
#     source：调用方刚核验的文件指纹；未提供时由本函数计算。
# 输出：字典，各字段如下。
#     sample：建模样本表；未启用时为 None。
#     sample_report：当前建模样本的抽样说明；未启用建模抽样时为 None。
#     audit：全量审计字典；未启用时为 None。
#     balance：全量协变量平衡结果表；未启用时为 None。
#     c2st_sample：两组等量的 C2ST 样本表；未启用时为 None。
#     srm：全量分组人数与公开参考比例的检查结果；未启用时为 None。
#     metadata：本次扫描的数据来源、全量人数、任务开关和抽样说明。
#     source：扫描使用的原始文件指纹。
# 说明：
'''
    四项开关任务说明：
    1. sample_size ：
        sample_size 默认100万 如果reuse=True 则None。
        sample_size 非none 控制 model_sample_enabled 打开，扫描过程中按照size进行随机优先级蓄水池抽样
        sample_size =0 则默认全量读取 不进行抽样
    2. per_arm：
        per_arm 默认None，此时不进行C2ST诊断抽样
        per_arm 非0，控制 c2st_sample_enabled开，扫描过程中表示按照随机优先级蓄水池抽样从从T/C分别抽per_arm量的样本
    3. audit_enabled 默认打开，扫描过程中 直接迭代更新全量基础审计（含SMD）
    4. diagnostics_enabled 默认打开，扫描过程中将全量特征和treatment写入临时文件；
                                   扫描结束后进行协变量平衡检验（SMD KS ）、SRM、C2ST【只进行了抽样】

'''
#-----------------------------------------------------------------------------------------------------------------------
def scan_raw_data(
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
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite raw data at row offset {total}; no rows were dropped')
            if not np.isin(values[:, len(FEATURES):], [0, 1]).all():
                raise ValueError('Raw labels must be binary')
            ids = np.arange(total, total + len(chunk), dtype=np.int64)
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_features —— 检查并提取特征。
# 输入：
#     df：原始数据表。
#     features：需要提取的特征名序列。
# 输出：
#     values：二维 NumPy 特征数组，列顺序与输入的特征名一致。
# 说明：确认特征数据可用，再整理成后续模型能够使用的数组。
#-----------------------------------------------------------------------------------------------------------------------
def _features(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    names = list(features)
    if not names or len(names) != len(set(names)):
        raise ValueError("features must be a nonempty sequence without duplicates")
    invalid = set(names) - _PRETREATMENT_FEATURES
    if invalid:
        raise ValueError(f"Only pre-treatment f0--f11 are allowed: {sorted(invalid)}")
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    values = df.loc[:, names].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite; handle missing values explicitly")
    return values


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_binary —— 检查并提取二分类标签。
# 输入：
#     df：原始数据表。
#     column：需要提取的标签列名。
# 输出：
#     标签数组：一维 NumPy 数组，每个值为 0 或 1。
# 说明：确认标签列符合二分类要求，再提取出来供后续检查或建模使用。
#-----------------------------------------------------------------------------------------------------------------------
def _binary(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        raise ValueError(f"Missing column: {column}")
    values = df[column].to_numpy(copy=True)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{column} must contain only binary 0/1 values")
    return values.astype(np.int8)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：srm_summary —— 检查两组人数与参考比例是否一致。
# 输入：
#     n1：Treatment 人数。
#     n0：Control 人数。
#     expected_ratio：参考 Treatment 占比，默认 0.85。
# 输出：字典，各字段如下。
#     n_treatment：Treatment 人数。
#     n_control：Control 人数。
#     observed_ratio：实际 Treatment 占比。
#     expected_ratio：用于检查的参考占比。
#     chi2：卡方检验统计量。
#     pvalue：检验的 p 值。
#     status：计算状态。
#     reason：结果解释。
# 说明：检查公开数据的分组人数是否明显偏离参考比例；结果不能单独证明随机化成立。
#-----------------------------------------------------------------------------------------------------------------------
def srm_summary(
    n1: int,
    n0: int,
    expected_ratio: float = 0.85,
) -> dict:
    if any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0 # 布尔值属于整数子类，需要显式排除。
        for n in (n1, n0)
    ):
        raise ValueError("Arm counts must be nonnegative integers")

    n = int(n1 + n0)
    if n == 0:
        raise ValueError("At least one observation is required")

    if (
        expected_ratio is None
        or not np.isfinite(expected_ratio)
        or not 0 < expected_ratio < 1
    ):
        raise ValueError(
            "expected_ratio must be a finite reference share strictly between 0 and 1"
        )

    expected1, expected0 = (n * expected_ratio,n * (1 - expected_ratio))
    test = chisquare(
        f_obs=[n1,n0],
        f_exp=[expected1,expected0],
    )
    return {
        "n_treatment": int(n1),
        "n_control": int(n0),
        "observed_ratio": float(n1 / n),
        "expected_ratio": float(expected_ratio),
        "chi2": float(test.statistic),
        "pvalue": float(test.pvalue),
        "status": "computed",
        "reason": (
            "Pearson chi-square count check against the supplied reference ratio; "
            "this alone does not validate original experiment randomization."
            + (
                " Expected arm count below 5; asymptotic p-value may be inaccurate."
                if min(expected1, expected0) < 5
                else ""
            )
        ),
    }


#-----------------------------------------------------------------------------------------------------------------------
# 函数：covariate_balance —— 全量SMD KS。
# 输入：
#     df：包含分组标签和特征的数据表。
#     features：需要检查的特征名序列。
# 输出：DataFrame，每个特征一行，各列如下。
#     feature：特征名称。
#     n_treatment：Treatment 人数。
#     n_control：Control 人数。
#     mean_treatment：Treatment 组的特征均值。
#     mean_control：Control 组的特征均值。
#     smd：标准化均值差。
#     abs_smd：标准化均值差的绝对值。
#     ks_statistic：KS 检验统计量。
#     ks_pvalue：KS 检验的 p 值。
# 说明：逐个特征比较两组的平均水平和分布，帮助发现可观测的不平衡。
#-----------------------------------------------------------------------------------------------------------------------
def covariate_balance(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    x = _features(df, features)
    treatment = _binary(df, "treatment")
    n1, n0 = int(treatment.sum()), int((treatment == 0).sum())
    if min(n1, n0) < 2:
        raise ValueError("Covariate balance needs at least two rows in each arm")
    records = []

    for j, feature in enumerate(features):
        treated, control = x[treatment == 1, j], x[treatment == 0, j]
        mean1, mean0 = float(treated.mean()), float(control.mean())
        scale = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        # 计算标准化均值差（SMD）。
        if scale > 0:
            smd = (mean1 - mean0) / scale
        else:
            smd = 0.0 if mean1 == mean0 else np.copysign(np.inf, mean1 - mean0)
        # 计算两样本 KS 统计量及 p 值。
        ks = ks_2samp(treated, control, alternative="two-sided", method="asymp")

        records.append(
            {
                "feature": feature,
                "n_treatment": n1,
                "n_control": n0,
                "mean_treatment": mean1,
                "mean_control": mean0,
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return pd.DataFrame.from_records(records)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_model —— 创建分类器。
# 输入：
#     name：模型名称，可选 logistic 或 xgboost。
#     seed：随机种子。
# 输出：
#     模型对象：尚未训练的 Logistic 流水线或 XGBoost 分类器。
# 说明：按指定名称准备一个新模型，供后续训练和预测使用。
#-----------------------------------------------------------------------------------------------------------------------
def _model(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, random_state=seed),
        )
    if name == "xgboost":
        # 沿用 30 轮、最多 15 叶的复杂度预算；学习率 0.1、L2=1。
        # 单线程避免大量置换拟合时过度订阅；配置预先固定，不根据留出 AUC 或 P 值调参。
        # min_child_weight 是 Hessian 权重阈值，不等同于 HGB 的最少叶样本数。
        from .runtime import configure_runtime
        configure_runtime()
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=30,
            max_depth=0,
            max_leaves=15,
            grow_policy="lossguide",
            min_child_weight=1.0,
            learning_rate=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_probability_metrics —— 计算预测评价指标。
# 输入：
#     y：真实的二分类标签数组。
#     prediction：每个样本预测为 1 的概率数组。
# 输出：字典，各字段如下。
#     roc_auc：ROC AUC 指标。
#     average_precision：平均精确率 AP 指标。
#     log_loss：对数损失指标。
#     brier：Brier 分数。
# 说明：将预测概率与真实标签进行比较，从不同角度评价模型的预测表现。
#-----------------------------------------------------------------------------------------------------------------------
def _probability_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    # y：真实的二分类标签。
    # prediction：每一行预测为 1 的概率。

    both_classes = np.unique(y).size == 2
    return {
        "roc_auc": float(roc_auc_score(y, prediction)) if both_classes else np.nan,
        "average_precision": (
            float(average_precision_score(y, prediction)) if np.any(y == 1) else np.nan
        ),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prediction)),
    }


#-----------------------------------------------------------------------------------------------------------------------
# 函数：c2st —— 检查特征能否预测分组。
# 输入：
#     df：包含分组标签和特征的诊断数据表。
#     features：用于预测的特征名序列。
#     sample_size：使用的样本数上限，默认 20,000。
#     permutations：置换次数，函数默认 99，主流程默认传入 499。
#     seed：随机种子，默认 42。
# 输出：字典，各字段如下。
#     summary：两个模型的真实预测指标及检验结果表。
#     null_auc：每个模型每次置换得到的 AUC 明细表。
#     sample_n：实际使用的总样本数。
#     train_n：内部训练样本数。
#     test_n：内部留出样本数。
#     seed：本次使用的随机种子。
#     interpretation：结果解释与适用限制。
# 说明：训练分类器区分两组，并与打乱标签后重新训练的表现比较，判断分组信息是否可预测。
#       同时校正两个模型的检验结果；诊断结果不能单独证明随机化是否成立。
#-----------------------------------------------------------------------------------------------------------------------
def c2st(
    df: pd.DataFrame, # 主流程通常传入已抽好的 20,000 行，T/C 为 1:1。
    features: Sequence[str],
    sample_size: int = 20_000,
    permutations: int = 99,
    seed: int = 42,
) -> dict:
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, (int, np.integer))
        or sample_size < 4
    ):
        raise ValueError("sample_size must be an integer of at least 4")
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, (int, np.integer))
        or permutations < 1
    ):
        raise ValueError("permutations must be a positive integer")


    rng = np.random.default_rng(seed)
    positions = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sample = df.iloc[positions]
    x = _features(sample, features)
    t = _binary(sample, "treatment")

    if len(t) < 4 or np.min(np.bincount(t, minlength=2)) < 2:
        raise ValueError("C2ST needs both treatment arms represented in train and test")

    train_idx, test_idx = train_test_split(
        np.arange(len(t)), test_size=0.3, stratify=t, random_state=seed
    )
    x_train, x_test = x[train_idx], x[test_idx]
    t_train, t_test = t[train_idx], t[test_idx]
    if min(np.unique(t_train).size, np.unique(t_test).size) != 2:
        raise ValueError("C2ST split has a single arm; supply a larger sample")
    
    # 常数基线：每个留出样本都预测训练集的 Treatment 占比；本次等量样本中为 0.5。
    dummy = _probability_metrics(t_test, np.full(len(t_test), t_train.mean()))
    
    # 用真实标签分别训练 Logistic 和 XGBoost，记录留出集的观测指标。
    observed = {}
    for name in ("logistic", "xgboost"):
        classifier = _model(name, seed)
        classifier.fit(x_train, t_train)
        observed[name] = _probability_metrics(
            t_test, classifier.predict_proba(x_test)[:, 1]
        )
    # 在训练集与留出集内分别打乱标签，每轮重新拟合，构造置换零分布。
    null_records = []
    for b in range(permutations):
        shuffled_train = rng.permutation(t_train)
        shuffled_test = rng.permutation(t_test)
        for name in ("logistic", "xgboost"):
            classifier = _model(name, seed)
            classifier.fit(x_train, shuffled_train)
            null_records.append(
                {
                    "model": name,
                    "permutation": b + 1,
                    "roc_auc": float(
                        roc_auc_score(
                            shuffled_test, classifier.predict_proba(x_test)[:, 1]
                        )
                    ),
                }
            )
    # null_auc：将每个模型、每次置换对应的 AUC 整理为结果表。
    null_auc = pd.DataFrame.from_records(null_records)

    records = []
    for name, metrics in observed.items():
        null = null_auc.loc[null_auc["model"] == name, "roc_auc"].to_numpy()
        # 上尾检验：统计零分布中 AUC 至少与观测值一样高的次数，计算置换 p 值。
        pvalue = (1 + np.count_nonzero(null >= metrics["roc_auc"])) / (permutations + 1)
        records.append(
            {
                "model": name,
                **metrics,
                "dummy_log_loss": dummy["log_loss"],
                "dummy_roc_auc": dummy["roc_auc"],
                "dummy_brier": dummy["brier"],
                "train_treatment_ratio": float(t_train.mean()),
                "test_treatment_ratio": float(t_test.mean()),
                "pvalue": float(pvalue),
                "null_auc_mean": float(null.mean()),
                "permutations": int(permutations),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary["pvalue_holm"] = multipletests(summary["pvalue"], method="holm")[1]
    return {
        "summary": summary,
        "null_auc": null_auc,
        "sample_n": len(t),
        "train_n": len(t_train),
        "test_n": len(t_test),
        "seed": seed,
        "interpretation": (
            "Exploratory check of observable assignment predictability on development data. "
            "A nonsignificant result does not prove randomization. A significant result "
            "requires investigation of design, sampling, and feature timing. "
            "Permutation exchangeability assumes an unstratified random assignment mechanism; "
            "unknown blocking or multiple experiments may require a different permutation scheme."
        ),
    }


#-----------------------------------------------------------------------------------------------------------------------
# 函数：learnability —— 检查特征能否预测转化或访问。
# 输入：
#     train：训练数据表。
#     val：验证数据表。
#     features：用于预测的特征名序列。
#     outcome：预测目标，可选 conversion 或 visit。
#     by_arm：是否同时分别在两组内训练和评价，默认 True。
#     seed：随机种子，默认 42。
# 输出：DataFrame，每种数据范围、每个模型一行，各列如下。
#     scope：评价的数据范围。
#     outcome：本次预测的目标。
#     n_train：对应训练样本数。
#     n_validation：对应验证样本数。
#     positive_rate_train：训练数据的正例比例。
#     positive_rate_validation：验证数据的正例比例。
#     model：模型名称。
#     status：计算或跳过的状态。
#     roc_auc：计算成功时记录的 ROC AUC 指标。
#     average_precision：计算成功时记录的 AP 指标。
#     log_loss：计算成功时记录的对数损失指标。
#     brier：计算成功时记录的 Brier 分数。
# 说明：比较模型与常数预测基线在验证数据上的表现，观察特征是否包含结果预测信息。
#       能预测结果不等于能预测处理效应；所有模型均跳过时，结果表不包含指标列。
#-----------------------------------------------------------------------------------------------------------------------
def learnability(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    outcome: str,
    by_arm: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    if outcome not in {"conversion", "visit"}:
        raise ValueError("outcome must be conversion or visit")
    x_train, x_val = _features(train, features), _features(val, features)
    y_train, y_val = _binary(train, outcome), _binary(val, outcome)
    t_train, t_val = _binary(train, "treatment"), _binary(val, "treatment")
    scopes = [("whole", np.ones(len(train), dtype=bool), np.ones(len(val), dtype=bool))]
    if by_arm:
        scopes.extend(
            [
                ("treatment", t_train == 1, t_val == 1),
                ("control", t_train == 0, t_val == 0),
            ]
        )
    records = []
    for scope, train_mask, val_mask in scopes:
        xt, xv = x_train[train_mask], x_val[val_mask]
        yt, yv = y_train[train_mask], y_val[val_mask]
        metadata = {
            "scope": scope,
            "outcome": outcome,
            "n_train": len(yt),
            "n_validation": len(yv),
            "positive_rate_train": float(yt.mean()) if len(yt) else np.nan,
            "positive_rate_validation": float(yv.mean()) if len(yv) else np.nan,
        }
        for name in ("dummy", "logistic", "xgboost"):
            record = {**metadata, "model": name}
            if len(yt) == 0 or len(yv) == 0:
                records.append({**record, "status": "skipped_empty_scope"})
                continue
            if name == "dummy":
                prediction = np.full(len(yv), yt.mean())
            elif np.unique(yt).size < 2:
                records.append({**record, "status": "skipped_single_class_train"})
                continue
            else:
                classifier = _model(name, seed)
                classifier.fit(xt, yt)
                prediction = classifier.predict_proba(xv)[:, 1]
            records.append(
                {**record, "status": "computed", **_probability_metrics(yv, prediction)}
            )
    return pd.DataFrame.from_records(records)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：run_learnability_diagnostics —— 检查特征对访问和转化的预测信息。
# 输入：
#     train：训练集数据表。
#     validation：验证集数据表，用于评价和策略选择。
#     out：主流程已创建的诊断结果目录。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     返回值：None；保存 learnability.csv。
# 说明：分别检查特征对访问和转化的预测能力，在验证集评价并保存两组结果。
#-----------------------------------------------------------------------------------------------------------------------
def run_learnability_diagnostics(train, validation, out, *, seed):
    records = [
        learnability(train, validation, FEATURES, outcome, seed=seed)
        for outcome in ("visit", "conversion")
    ]
    out = safe_output(out)
    write_result(pd.concat(records), out / 'learnability.csv')


#-----------------------------------------------------------------------------------------------------------------------
# 函数：update_experiment_diagnostics —— 组织诊断并保存结果。
# 输入：
#     path：原始 CSV 文件路径。
#     out：调用方已创建的诊断结果目录。
#     audit：当前实验记录的审计人数信息。
#     source：当前实验记录的数据来源信息。
#     sample_size：C2ST 使用的总样本数，默认 20,000。
#     permutations：置换次数，默认 499。
#     seed：随机种子，默认 42。
#     progress：接收进度文字的函数，默认 print。
#     scanned：统一扫描已返回的结果；未提供时只扫描诊断所需的数据。
# 输出：
#     details：返回的诊断详情字典。
# 保存文件：
#     balance.csv：全量特征平衡结果。
#     srm.json：分组人数比例检查结果。
#     c2st.csv：两个分类器的预测指标和检验结果。
#     c2st_null.csv：每轮置换的 AUC 明细。
#     c2st_details.json：与返回的 details 对应的诊断详情。
# 说明：复用统一扫描的平衡检查、SRM 和诊断样本，运行 C2ST 后保存结果。
#       单独更新诊断时自行调用统一扫描；主流程传入已有结果，不再扫描 CSV。
#-----------------------------------------------------------------------------------------------------------------------
def update_experiment_diagnostics(
    path, out, *, audit, source, sample_size=20_000, permutations=499,
    seed=42, progress=print, scanned=None,
):
    if sample_size < 20 or sample_size % 20:
        raise ValueError("C2ST sample_size must be >=20 and divisible by 20")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    out = safe_output(out)
    if not out.is_dir():
        raise FileNotFoundError(out)
    started = time.perf_counter()
    if scanned is None:
        scanned = scan_raw_data(
            path, sample_size=None, per_arm=sample_size // 2, seed=seed, progress=progress,
            audit_enabled=False, diagnostics_enabled=True,
        )
    balance, sample, metadata = scanned['balance'], scanned['c2st_sample'], scanned['metadata']
    if metadata is None or balance is None or sample is None or scanned['srm'] is None:
        raise ValueError('扫描结果未包含完整诊断数据')
    if metadata['seed'] != seed or metadata['sample_n'] != sample_size:
        raise ValueError('扫描结果的 C2ST 样本规模或种子与当前参数不同')
    if metadata["source"]["sha256"] != source["sha256"]:
        raise ValueError("诊断与当前实验的原始数据版本不同")
    for key in ("n_treatment", "n_control"):
        if metadata[key] != audit[key]:
            raise ValueError(f"全量诊断计数与 audit 不一致：{key}")
    from threadpoolctl import threadpool_limits

    progress(f"全量来源 C2ST：两组各 {sample_size // 2:,} 行，{permutations} 次完整重拟合置换")
    # 将本次诊断使用的线程数限制为 4，控制完整重拟合置换的计算资源。
    with threadpool_limits(limits=4):
        check = c2st(sample, FEATURES, sample_size, permutations, seed)
    summary = check["summary"].assign(
        scope="full_source_arm_balanced_sample", representation="raw_csv_float64",
        metric="ROC AUC", sample_n=sample_size,
        n_control=sample_size // 2, n_treatment=sample_size // 2,
        pvalue_resolution=1 / (permutations + 1),
        source_sha256=metadata["source"]["sha256"],
    )
    if not (summary[["train_treatment_ratio", "test_treatment_ratio"]] == .5).all().all():
        raise AssertionError("C2ST 内部两组人数不平衡")
    details = {
        **{key: value for key, value in check.items() if not isinstance(value, pd.DataFrame)},
        "scope": "full_source_arm_balanced_sample", "source": metadata["source"],
        "representation": "raw_csv_float64", "features": FEATURES,
        "full_n": metadata["n"], "full_n_control": metadata["n_control"],
        "full_n_treatment": metadata["n_treatment"], "rows_dropped": metadata["rows_dropped"],
        "sampling": metadata["sampling"], "sampling_target": metadata["sampling_target"],
        "n_control": sample_size // 2, "n_treatment": sample_size // 2,
        "train_n_control": check["train_n"] // 2, "train_n_treatment": check["train_n"] // 2,
        "holdout_n_control": check["test_n"] // 2, "holdout_n_treatment": check["test_n"] // 2,
        "metric": "one-sided ROC AUC", "multiplicity": "Holm across two prespecified classifiers",
        "models": ["logistic", "xgboost"],
        "xgboost_parameters": _model("xgboost", seed).get_params(),
        "permutations": permutations, "pvalue_resolution": 1 / (permutations + 1),
        "sample_row_ids_sha256": hashlib.sha256(sample.row_id.to_numpy().tobytes()).hexdigest(),
        "ks_method": metadata["ks_method"], "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "全量原始数据组内等概率、两组等量抽样，比较P(X|T=1)与P(X|T=0)。"
            "不显著不证明联合分布相同或随机化；显著不单独证明随机化失败。"
            "置换依赖可交换性，未知分块、多实验和实体依赖未验证。"
            "这是事后描述性诊断，不用于正式模型选择。"
        ),
    }
    # 所有计算及来源核对完成后，直接写入调用方传入的诊断目录。
    write_result(balance, out / 'balance.csv')
    write_result(summary, out / 'c2st.csv')
    write_result(check["null_auc"].assign(scope=details["scope"]), out / 'c2st_null.csv')
    write_result(details, (out / 'c2st_details.json'))
    write_result(scanned['srm'], (out / 'srm.json'))
    return details
