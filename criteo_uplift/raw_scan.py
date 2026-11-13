"""Validate, audit and sample the source CSV in a single read."""

from contextlib import ExitStack
import hashlib
from io import BufferedReader, FileIO
from os import fstat
from pathlib import Path
import re
from tempfile import TemporaryFile

import numpy as np
import pandas as pd

from .config import COLUMNS, DEFAULT_DATA, FEATURES, LABELS
from .input_checks import check_binary_vector, check_feature_matrix
from .diagnostics import covariate_balance, srm_summary


_CSV_RECORDS = re.compile(rb'(?:(?:[^,\n]*,){15}[^,\n]*\n)+')

def validate_chunk(chunk, offset):
    #-------------
    # 校验当前块的全部原始行，返回共用数值视图、原始行号及处理分组。
    # 输入参数：
    #     - chunk：当前经过验证的原始数据块，所有原始列保持 float64。
    #     - offset：当前块在原始 CSV 数据行中的起始位置。
    # 输出参数：
    #     - (原始数值视图, 原始行号, 处理分组向量)；非法行会抛出 ValueError。
    #-------------
    values = chunk.to_numpy()
    check_feature_matrix(values)
    for label in LABELS:
        check_binary_vector(chunk[label], label, len(chunk))
    ids = np.arange(offset, offset + len(chunk), dtype=np.int64)
    return values, ids, chunk.treatment.to_numpy(dtype=np.int8)


class _SourceFile(FileIO):
    """Hash bytes and check record width as pandas consumes the source."""

    def __init__(self, path):
        super().__init__(path, mode='rb')
        self.digest = hashlib.sha256()
        self.partial_row = b''

    def readinto(self, buffer):
        count = super().readinto(buffer)
        if count:
            block = memoryview(buffer)[:count]
            self.digest.update(block)
            records = self.partial_row + block.tobytes()
            end = records.rfind(b'\n') + 1
            if end and not _CSV_RECORDS.fullmatch(records, 0, end):
                raise ValueError('原始 CSV 每行须含 16 个字段')
            self.partial_row = records[end:]
        elif self.partial_row:
            if not _CSV_RECORDS.fullmatch(self.partial_row + b'\n'):
                raise ValueError('原始 CSV 每行须含 16 个字段')
            self.partial_row = b''
        return count


def _file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def scan_raw_data(
    path=DEFAULT_DATA, sample_size=1_000_000, seed=42, chunk_size=250_000,
    *, per_arm=10_000, progress=print,
):
    """Validate every source row and compute the audit and both samples."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError('sample_size 必须为大于等于 0 的整数')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size 必须为正整数')
    if isinstance(per_arm, bool) or not isinstance(per_arm, int) or per_arm < 2:
        raise ValueError('per_arm 必须为大于等于 2 的整数')

    path = Path(path).resolve()
    model = ModelSampler(sample_size, seed)
    audit = AuditAccumulator()
    total, counts = 0, np.zeros(2, dtype=np.int64)
    with ExitStack() as stack:
        diagnostics = DiagnosticsAccumulator(stack, per_arm=per_arm, seed=seed)
        raw = stack.enter_context(_SourceFile(path))
        before = fstat(raw.fileno())
        stream = stack.enter_context(BufferedReader(raw))
        reader = stack.enter_context(pd.read_csv(stream, dtype=np.float64, chunksize=chunk_size))
        for chunk in reader:
            if list(chunk.columns) != COLUMNS or not isinstance(chunk.index, pd.RangeIndex):
                raise ValueError(f'CSV 列名及顺序必须严格为 {COLUMNS}，且每行须含 16 个字段')
            if chunk.empty:
                continue
            values, ids, treatment = validate_chunk(chunk, total)
            counts += np.bincount(treatment, minlength=2)
            audit.update(chunk, values, treatment)
            model.update(chunk, values, ids)
            diagnostics.update(chunk, treatment, ids)
            total += len(chunk)
            if total % 1_000_000 == 0:
                progress(f'Raw scan: {total:,} rows')

        if (_file_identity(fstat(raw.fileno())) != _file_identity(before)
                or _file_identity(path.stat()) != _file_identity(before)):
            raise RuntimeError('原始文件在扫描期间发生变化')
        source = {
            'path': str(path), 'size': before.st_size, 'mtime_ns': before.st_mtime_ns,
            'sha256': raw.digest.hexdigest(),
        }
        if total == 0 or (counts == 0).any():
            raise ValueError('数据为空或缺少 treatment/control')
        if min(counts) < per_arm:
            raise ValueError(f'每个处理臂至少需要 {per_arm} 行；实际人数为 {counts.tolist()}')

        progress(f'Raw scan: n={total:,}; control={counts[0]:,}; treatment={counts[1]:,}')
        balance, c2st_sample, srm, feature_stats = diagnostics.finish(total, counts, source, progress)
        sample = model.finish()
        return {
            'sample': sample, 'source': source,
            'sample_report': sample_report(sample, sample_size=sample_size, seed=seed,
                                           source=source, total=total),
            'audit': audit.finish(total, counts, feature_stats),
            'balance': balance, 'c2st_sample': c2st_sample, 'srm': srm,
            'metadata': {
                'source': source, 'n': total, 'n_treatment': int(counts[1]),
                'n_control': int(counts[0]), 'rows_dropped': 0, 'seed': seed,
                'sample_n': len(c2st_sample),
                'sampling': 'independent uniform-priority reservoir without replacement within each arm',
                'sampling_target': 'P(X|T=1) versus P(X|T=0); balanced arms',
                'ks_method': 'two-sided asymptotic; unadjusted exploratory p-values',
            },
        }


class ModelSampler:
    #-------------
    # 建模样本池：全量行统一随机优先级，最终特征保存为 float32。
    #-------------
    def __init__(self, sample_size, seed):
        #-------------
        # 初始化独立随机流、建模样本池及原始行号；仅全量模式保留所有块。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - sample_size：建模目标样本行数；0 表示保留全部原始行。
        #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
        # 输出参数：
        #     - 无返回值；初始化独立随机流及建模样本池。
        #-------------
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.reservoir = np.empty((0, len(COLUMNS)), dtype=np.float64)
        self.row_ids = np.empty(0, dtype=np.int64)
        self.priorities = np.empty(0)
        self.all_chunks = []

    def update(self, chunk, values, ids):
        #-------------
        # 为当前块分配随机优先级，更新样本池；全量模式直接保留建模表示。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - chunk：当前经过验证的原始数据块，所有原始列保持 float64。
        #     - values：当前数据块的二维 float64 数值视图，与读取器共享底层数据。
        #     - ids：当前数据块的原始位置行号，为从 0 开始的 int64 向量。
        # 输出参数：
        #     - 无返回值；原位更新样本池、优先级和来源行号。
        #-------------
        if self.sample_size == 0:
            model_chunk = chunk.astype({
                **{f: 'float32' for f in FEATURES},
                **{label: 'int8' for label in LABELS},
            })
            model_chunk.insert(0, 'row_id', ids)
            self.all_chunks.append(model_chunk)
        else:
            self.reservoir = np.concatenate([self.reservoir, values])
            self.priorities = np.concatenate([self.priorities, self.rng.random(len(chunk))])
            self.row_ids = np.concatenate([self.row_ids, ids])
            if len(self.reservoir) > self.sample_size:
                keep = np.argpartition(self.priorities, self.sample_size - 1)[:self.sample_size]
                self.reservoir, self.priorities, self.row_ids = self.reservoir[keep], self.priorities[keep], self.row_ids[keep]

    def finish(self):
        #-------------
        # 按原始行号排序并生成建模样本，特征转为 float32、标签转为 int8。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        # 输出参数：
        #     - 按原始行号排序的建模 DataFrame，特征为 float32，标签为 int8。
        #-------------
        if self.sample_size == 0:
            sample = pd.concat(self.all_chunks, ignore_index=True)
        else:
            order = np.argsort(self.row_ids)
            sample = pd.DataFrame(self.reservoir[order], columns=COLUMNS).astype({
                **{f: 'float32' for f in FEATURES}, **{label: 'int8' for label in LABELS},
            })
            sample.insert(0, 'row_id', self.row_ids[order])
        return sample


class BalancedSampler:
    #-------------
    # 诊断样本池：两臂分别维护随机流和样本池，不消耗建模抽样的随机数。
    #-------------
    def __init__(self, per_arm, seed):
        #-------------
        # 从种子派生两个独立随机流，初始化处理组和对照组样本池。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - per_arm：Diagnostics 中每个处理臂的 C2ST 目标样本行数。
        #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
        # 输出参数：
        #     - 无返回值；初始化两个独立随机流和两臂样本池。
        #-------------
        self.per_arm = per_arm
        self.generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        self.diagnostic_reservoirs = [pd.DataFrame(), pd.DataFrame()]
        self.diagnostic_keys = [np.empty(0), np.empty(0)]

    def update(self, chunk, treatment, ids):
        #-------------
        # 分别向两臂样本池补入当前块候选行，保持 float64 特征精度。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - chunk：当前经过验证的原始数据块，所有原始列保持 float64。
        #     - treatment：当前块的处理分组向量，0 为对照组，1 为处理组。
        #     - ids：当前数据块的原始位置行号，为从 0 开始的 int64 向量。
        # 输出参数：
        #     - 无返回值；分别更新两臂候选样本及优先级。
        #-------------
        for arm in (0, 1):
            mask = treatment == arm
            arm_rows = chunk.loc[mask]
            arm_rows['row_id'] = ids[mask]
            combined_keys = np.concatenate([
                self.diagnostic_keys[arm], self.generators[arm].random(len(arm_rows)),
            ])
            combined = pd.concat([self.diagnostic_reservoirs[arm], arm_rows], ignore_index=True)
            if len(combined) > self.per_arm:
                keep = np.argpartition(combined_keys, self.per_arm - 1)[:self.per_arm]
                combined, combined_keys = combined.iloc[keep], combined_keys[keep]
            self.diagnostic_reservoirs[arm], self.diagnostic_keys[arm] = combined, combined_keys

    def finish(self):
        #-------------
        # 合并两臂等量样本，按原始行号排序并校验数量与唯一性。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        # 输出参数：
        #     - 按原始行号排序的两臂等量 DataFrame，特征保持 float64。
        #-------------
        c2st_sample = pd.concat(self.diagnostic_reservoirs, ignore_index=True).sort_values('row_id').reset_index(drop=True)
        for label in LABELS:
            c2st_sample[label] = c2st_sample[label].astype(np.int8)
        c2st_sample['row_id'] = c2st_sample.row_id.astype(np.int64)
        if not c2st_sample.row_id.is_unique or len(c2st_sample) != 2 * self.per_arm:
            raise AssertionError('诊断样本池的行号或样本数量不符合要求')
        return c2st_sample


def sample_report(sample, *, sample_size, seed, source, total):
    #-------------
    # 根据最终建模样本及扫描总行数生成抽样说明，不依赖审计任务。
    # 输入参数：
    #     - sample：包含原始行号、float32 特征和二元标签的建模样本。
    #     - sample_size：建模目标样本行数；0 表示保留全部原始行。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    #     - total：本次扫描读取的原始数据总行数。
    # 输出参数：
    #     - 建模样本报告字典，包含来源规模、抽样参数、数据表示和分组汇总。
    #-------------
    sample_report = {
        'scope': 'modeling_sample',
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
        # 同时输出抽样分组汇总。
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
    return sample_report


class AuditAccumulator:
    #-------------
    # 全量审计累计器：重复行指纹、标签异常计数及稳定的分组矩统计。
    #-------------
    def __init__(self):
        self.hashes = []
        self.positives = np.zeros((2, 3), dtype=np.int64)
        self.bad_visit = 0
        self.control_exposure = 0

    def update(self, chunk, values, treatment):
        self.hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        for arm in (0, 1):
            outcomes = values[treatment == arm, len(FEATURES) + 1:]
            self.positives[arm] += outcomes.sum(axis=0).astype(np.int64)
        self.bad_visit += int(((chunk.conversion == 1) & (chunk.visit == 0)).sum())
        self.control_exposure += int(((chunk.treatment == 0) & (chunk.exposure == 1)).sum())

    def finish(self, total, counts, feature_stats):
        duplicate_excess = total - int(np.unique(np.concatenate(self.hashes)).size)
        #输出完整审计字典
        audit = {
            'n_rows': total, 'n_columns': len(COLUMNS),
            'n_treatment': int(counts[1]), 'n_control': int(counts[0]),
            'treatment_ratio': counts[1] / total,
            'missing_or_nonfinite': 0, 'invalid_binary_labels': 0,
            'control_exposed': self.control_exposure, 'conversion_without_visit': self.bad_visit,
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
                    **{label: {'positive': self.positives[arm, k], 'rate': self.positives[arm, k] / counts[arm]}
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
                    ('Control', counts[0], self.positives[0, 0]),
                    ('Treatment', counts[1], self.positives[1, 0]),
                    ('Total', total, self.positives[:, 0].sum()),
                )
            ],
        }
        return audit



class DiagnosticsAccumulator:
    #-------------
    # 全量诊断累计器：临时列文件支持精确 KS，独立样本池支持 C2ST。
    #-------------
    def __init__(self, stack, *, per_arm, seed):
        #-------------
        # 按需建立全量特征临时文件和两臂等量样本池，由外层统一管理文件关闭。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - stack：外层 ExitStack，统一管理诊断临时文件的生命周期。
        #     - per_arm：Diagnostics 中每个处理臂的 C2ST 目标样本行数。
        #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
        # 输出参数：
        #     - 无返回值；创建全量列临时文件和两臂等量样本池。
        #-------------
        self.sampler = BalancedSampler(per_arm, seed)
        self.files = {
            name: stack.enter_context(TemporaryFile())
            for name in FEATURES + ['treatment']
        }

    def update(self, chunk, treatment, ids):
        #-------------
        # 写入当前块的 float64 特征和分组标签，并按需更新两臂样本池。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - chunk：当前经过验证的原始数据块，所有原始列保持 float64。
        #     - treatment：当前块的处理分组向量，0 为对照组，1 为处理组。
        #     - ids：当前数据块的原始位置行号，为从 0 开始的 int64 向量。
        # 输出参数：
        #     - 无返回值；向临时列文件追加数据并更新 C2ST 样本池。
        #-------------
        for feature in FEATURES:
            chunk[feature].to_numpy().tofile(self.files[feature])
        treatment.tofile(self.files['treatment'])
        self.sampler.update(chunk, treatment, ids)

    def finish(self, total, counts, source, progress):
        """Compute full-source balance and share feature summaries with the audit."""
        c2st_sample = self.sampler.finish()
        records, feature_stats = [], []
        for file in self.files.values():
            file.flush()
        treatment = np.memmap(self.files['treatment'], dtype=np.int8, mode='r', shape=(total,))
        for feature in FEATURES:
            values = np.memmap(self.files[feature], dtype=np.float64, mode='r', shape=(total,))
            frame = pd.DataFrame({feature: values, 'treatment': treatment}, copy=False)
            record = covariate_balance(frame, [feature]).iloc[0].to_dict()
            records.append(record)
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
        balance = pd.DataFrame(records).assign(
            scope='full_published_dataset', representation='raw_csv_float64', source_sha256=source['sha256'],
        )
        srm = {
            **srm_summary(int(counts[1]), int(counts[0]), expected_ratio=.85),
            'scope': 'full_published_dataset', 'source_sha256': source['sha256'],
            'reference_source': 'https://arxiv.org/html/2111.10106#S3',
            'interpretation': '公开数据分配计数与85%外部参考比例的一致性；不验证原始线上实验SRM或随机化。',
        }
        return balance, c2st_sample, srm, feature_stats
