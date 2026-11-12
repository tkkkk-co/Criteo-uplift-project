#-------------
# 通过单次只读 CSV 扫描完成数据校验，并管理各任务的状态和资源生命周期。
# 此文件不管理持久缓存；抽样、审计和诊断分别维护自己的累计量。
# 各任务共用读取器提供的 float64 数据块，不修改原始数据块。
#-------------
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
import pandas as pd

from .config import COLUMNS, DEFAULT_DATA, FEATURES, LABELS
from .input_checks import check_binary_vector, check_feature_matrix
from .pipeline_io import fingerprint
from .diagnostics import covariate_balance, srm_summary



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


def scan_raw_data(
    path=DEFAULT_DATA, sample_size=1_000_000, seed=42, chunk_size=250_000,
    *, per_arm=10_000, sample_enabled=True, audit_enabled=True, diagnostics_enabled=True,
    progress=print, source=None,
):
    #-------------
    # 功能：读取一次 CSV，将每个校验后的数据块分发给本次需要计算的任务。
    # 资源：ExitStack 统一关闭读取器与诊断临时文件，校验或计算失败时同样释放。
    # 输入参数：
    #     - path：原始 CSV 路径。
    #     - sample_size：建模目标样本行数；0 表示保留全部原始行。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    #     - chunk_size：每次解析原始 CSV 的行数，必须为正整数。
    #     - per_arm：Diagnostics 中每个处理臂的 C2ST 目标样本行数。
    #     - sample_enabled：是否启用建模抽样任务。
    #     - audit_enabled：是否启用全量审计任务。
    #     - diagnostics_enabled：是否启用全量诊断和两臂 C2ST 抽样任务。
    #     - progress：接收扫描进度文本的回调函数。
    #     - source：可选来源指纹；None 时在读取前计算，结束时再次核验。
    # 输出参数：
    #     - 包含建模样本、抽样说明、审计、平衡表、C2ST 样本、SRM、元数据及来源的字典。
    #-------------
    for name, enabled in (('sample_enabled', sample_enabled), ('audit_enabled', audit_enabled),
                          ('diagnostics_enabled', diagnostics_enabled)):
        if not isinstance(enabled, bool):
            raise ValueError(f'{name} 必须为布尔值')
    if not any((sample_enabled, audit_enabled, diagnostics_enabled)):
        raise ValueError('至少启用一个扫描任务')
    if sample_enabled and (
        isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0
    ):
        raise ValueError('sample_size 必须为大于等于 0 的整数')
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size 必须为正整数')
    if diagnostics_enabled and (
        isinstance(per_arm, bool) or not isinstance(per_arm, int) or per_arm < 2
    ):
        raise ValueError('per_arm 必须为大于等于 2 的整数')
    per_arm = per_arm if diagnostics_enabled else None
    path = Path(path)
    source = fingerprint(path) if source is None else source
    if list(pd.read_csv(path, nrows=0).columns) != COLUMNS:
        raise ValueError(f'CSV 列名及顺序必须严格为 {COLUMNS}')
    model = ModelSampler(sample_size, seed) if sample_enabled else None
    audit = AuditAccumulator(streaming_features=not diagnostics_enabled) if audit_enabled else None
    total, counts = 0, np.zeros(2, dtype=np.int64)
    with ExitStack() as stack:
        diagnostics = DiagnosticsAccumulator(
            stack, per_arm=per_arm, seed=seed,
        ) if diagnostics_enabled else None
        reader = stack.enter_context(pd.read_csv(path, dtype=np.float64, chunksize=chunk_size))
        for chunk in reader:
            values, ids, treatment = validate_chunk(chunk, total)
            for arm in (0, 1):
                counts[arm] += int((treatment == arm).sum())
            if audit is not None:
                audit.update(chunk, values, treatment, counts)
            if model is not None:
                model.update(chunk, values, ids)
            if diagnostics is not None:
                diagnostics.update(chunk, treatment, ids)
            total += len(chunk)
            if total % 1_000_000 == 0:
                progress(f'Raw scan: {total:,} rows')
        result = finalize_scan(
            model, audit, diagnostics, total, counts, source, seed, chunk_size, per_arm,
            diagnostics_enabled, progress,
        )
    if fingerprint(path) != source:
        raise RuntimeError('原始文件在扫描期间发生变化')
    return result


def finalize_scan(model, audit, diagnostics, total, counts, source, seed,
                  chunk_size, per_arm, diagnostics_enabled, progress):
    #-------------
    # 确认全量两臂齐全、诊断样本数量充足，再整理各任务结果及扫描元数据。
    # 输入参数：
    #     - model：建模样本池；未计算 Sample 时为 None。
    #     - audit：全量审计累计器；未计算 Audit 时为 None。
    #     - diagnostics：全量诊断累计器；未计算 Diagnostics 时为 None。
    #     - total：本次扫描读取的原始数据总行数。
    #     - counts：截至当前的全量两臂人数，依次为对照组、处理组。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    #     - chunk_size：每次解析原始 CSV 的行数，必须为正整数。
    #     - per_arm：Diagnostics 中每个处理臂的 C2ST 目标样本行数。
    #     - diagnostics_enabled：是否启用全量诊断和两臂 C2ST 抽样任务。
    #     - progress：接收扫描进度文本的回调函数。
    # 输出参数：
    #     - 各任务最终结果和来源说明组成的字典；缺少处理臂或样本不足时抛出 ValueError。
    #-------------
    if total == 0 or (counts == 0).any():
        raise ValueError('数据为空或缺少 treatment/control')
    if per_arm is not None and min(counts) < per_arm:
        raise ValueError(f'每个处理臂至少需要 {per_arm} 行；实际人数为 {counts.tolist()}')
    progress(f'Raw scan: n={total:,}; control={counts[0]:,}; treatment={counts[1]:,}')
    balance, c2st_sample, srm, feature_stats = (None,) * 4
    if diagnostics is not None:
        balance, c2st_sample, srm, feature_stats = diagnostics.finish(
            total, counts, source, progress, include_audit_stats=audit is not None,
        )
    sample = model.finish() if model is not None else None
    report = sample_report(sample, sample_size=model.sample_size, seed=seed, source=source, total=total) if model is not None else None
    return {
        'sample': sample, 'sample_report': report,
        'audit': audit.finish(total, counts, feature_stats) if audit is not None else None,
        'balance': balance, 'c2st_sample': c2st_sample, 'srm': srm, 'source': source,
        'metadata': scan_metadata(
            total=total, counts=counts, source=source, seed=seed, chunk_size=chunk_size,
            per_arm=per_arm, audit_enabled=audit is not None,
            diagnostics_enabled=diagnostics_enabled, sample_enabled=model is not None,
            c2st_sample=c2st_sample,
        ),
    }


def scan_metadata(*, total, counts, source, seed, chunk_size, per_arm, audit_enabled,
                  diagnostics_enabled, sample_enabled, c2st_sample):
    #-------------
    # 生成来源、全量人数、数据表示、任务开关和抽样方法说明。
    # 输入参数：
    #     - total：本次扫描读取的原始数据总行数。
    #     - counts：全量两臂人数；只复用建模缓存时未知的全量计数为 None。
    #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
    #     - seed：随机种子；建模抽样和两臂诊断抽样使用独立随机流。
    #     - chunk_size：每次解析原始 CSV 的行数，必须为正整数。
    #     - per_arm：Diagnostics 中每个处理臂的 C2ST 目标样本行数。
    #     - audit_enabled：是否启用全量审计任务。
    #     - diagnostics_enabled：是否启用全量诊断和两臂 C2ST 抽样任务。
    #     - sample_enabled：是否启用建模抽样任务。
    #     - c2st_sample：两臂等量的 float64 诊断样本；诊断关闭时为 None。
    # 输出参数：
    #     - 包含来源、全量人数、任务开关、执行状态及抽样方法的元数据字典。
    #-------------
    metadata = {
        'scope': 'full_published_dataset', 'source': source, 'representation': 'raw_csv_float64',
        'features': FEATURES, 'validated_columns': COLUMNS,
        'n': total, 'n_treatment': int(counts[1]) if counts[1] is not None else None,
        'n_control': int(counts[0]) if counts[0] is not None else None,
        'rows_dropped': 0, 'seed': seed, 'chunk_size': chunk_size,
        'audit_enabled': audit_enabled, 'diagnostics_enabled': diagnostics_enabled,
        'sample_enabled': sample_enabled, 'model_sample_enabled': sample_enabled,
        'c2st_sample_enabled': diagnostics_enabled,
        'task_status': {
            name: 'computed' if enabled else 'disabled'
            for name, enabled in (('sample', sample_enabled), ('audit', audit_enabled),
                                  ('diagnostics', diagnostics_enabled))
        },
        'sample_n': len(c2st_sample) if diagnostics_enabled else None,
        'sample_n_treatment': per_arm, 'sample_n_control': per_arm,
        'sampling': 'independent uniform-priority reservoir without replacement within each arm' if diagnostics_enabled else None,
        'sample_scope': 'full_source_arm_balanced_sample' if diagnostics_enabled else None,
        'sampling_target': 'P(X|T=1) versus P(X|T=0); balanced arms, not population T prevalence' if diagnostics_enabled else None,
        'row_id_definition': 'zero-based original CSV data-row position; not a user ID',
        'ks_method': 'two-sided asymptotic; unadjusted exploratory p-values; ties and dependence limit calibration' if diagnostics_enabled else None,
    }
    return metadata


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
    return sample_report


class AuditAccumulator:
    #-------------
    # 全量审计累计器：重复行指纹、标签异常计数及稳定的分组矩统计。
    #-------------
    def __init__(self, *, streaming_features):
        #-------------
        # 初始化审计累计量；仅在不执行全量诊断时累计流式特征统计。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - streaming_features：是否累计流式特征矩统计；全量诊断提供共享统计时为 False。
        # 输出参数：
        #     - 无返回值；初始化审计计数、行指纹列表和可选流式统计累计量。
        #-------------
        self.streaming_features = streaming_features
        self.hashes = []
        self.positives = np.zeros((2, 3), dtype=np.int64)
        self.means = np.zeros((2, len(FEATURES)))
        self.centered_squares = np.zeros_like(self.means)
        self.minimum = np.full(len(FEATURES), np.inf)
        self.maximum = np.full(len(FEATURES), -np.inf)
        self.bad_visit = 0
        self.control_exposure = 0

    def update(self, chunk, values, treatment, counts):
        #-------------
        # 累计全部行的重复指纹、标签正例和异常计数，按需合并特征均值及方差。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - chunk：当前经过验证的原始数据块，所有原始列保持 float64。
        #     - values：当前数据块的二维 float64 数值视图，与读取器共享底层数据。
        #     - treatment：当前块的处理分组向量，0 为对照组，1 为处理组。
        #     - counts：截至当前的全量两臂人数，依次为对照组、处理组。
        # 输出参数：
        #     - 无返回值；累计全量质量计数、重复指纹及可选的分组矩统计。
        #-------------
        x = values[:, :len(FEATURES)]
        self.hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        for arm in (0, 1):
            mask = treatment == arm #布尔数组，标记当前块中哪些行属于这一组
            self.positives[arm] += values[mask, len(FEATURES) + 1:].sum(axis=0).astype(np.int64)
            # 未启用完整平衡检查时，需要逐块累计特征统计；启用时程序会把全量特征写入临时文件 之后利用covariate_balance 计算均值 SMD KS
            if self.streaming_features:
                sub = x[mask]
                if len(sub):
                    # 合并旧数据，分别在T/C组下计算均值 方差 用于后续计算SMD
                    old_n = counts[arm] - len(sub)
                    chunk_mean = sub.mean(axis=0)
                    difference = chunk_mean - self.means[arm] #新旧两批数据的均值差，means[arm] 仍然是合并当前数据块之前的均值
                    #更新分组方差：分批合并方差统计量经典公式/算法，能够避免保存全部历史数据
                    self.centered_squares[arm] += (
                        np.square(sub - chunk_mean).sum(axis=0)
                        + np.square(difference) * old_n * (len(sub) / counts[arm])
                    )
                    #更新分组均值：
                    self.means[arm] += difference * (len(sub) / counts[arm])
        if self.streaming_features:
            self.minimum = np.minimum(self.minimum, x.min(axis=0))
            self.maximum = np.maximum(self.maximum, x.max(axis=0))
        self.bad_visit += int(((chunk.conversion == 1) & (chunk.visit == 0)).sum())
        self.control_exposure += int(((chunk.treatment == 0) & (chunk.exposure == 1)).sum())

    def _feature_stats(self, counts):
        #-------------
        # 将流式均值和中心平方和整理为全量分组标准差、范围及 SMD。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - counts：截至当前的全量两臂人数，依次为对照组、处理组。
        # 输出参数：
        #     - 按特征排列的记录列表，包含范围、两臂均值、标准差及 SMD。
        #-------------
        feature_stats = []
        for j, feature in enumerate(FEATURES):
            row = {'feature': feature, 'min': self.minimum[j], 'max': self.maximum[j]}
            for arm in (0, 1):
                row[f'mean_{arm}'] = self.means[arm, j]
                row[f'std_{arm}'] = (
                    # 偏差平方和除以自由度，再开方，得到样本标准差。
                    np.sqrt(max(0, self.centered_squares[arm, j] / (counts[arm] - 1)))
                    if counts[arm] > 1 else np.nan
                )
            denominator = np.sqrt((row['std_0'] ** 2 + row['std_1'] ** 2) / 2)
            difference = row['mean_1'] - row['mean_0']
            row['smd'] = difference / denominator if denominator > 0 else (
                0.0 if denominator == 0 and difference == 0 else np.nan
            )
            row['smd_status'] = 'computed' if np.isfinite(row['smd']) else 'undefined_zero_or_missing_variance'
            feature_stats.append(row)
        return feature_stats

    def finish(self, total, counts, feature_stats=None):
        #-------------
        # 结合流式或诊断共享的特征统计，生成完整全量审计结果。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - total：本次扫描读取的原始数据总行数。
        #     - counts：截至当前的全量两臂人数，依次为对照组、处理组。
        #     - feature_stats：可复用的全量特征描述统计；为 None 时使用本任务的流式统计。
        # 输出参数：
        #     - 全量审计字典，包含重复行、逻辑异常、分组计数和特征统计。
        #-------------
        if feature_stats is None:
            feature_stats = self._feature_stats(counts)
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

    def finish(self, total, counts, source, progress, *, include_audit_stats):
        #-------------
        # 逐列读取全量特征完成诊断，并按需向审计提供可复用的描述性统计。
        # 输入参数：
        #     - self：当前实例，保存本任务的累计状态或测试上下文。
        #     - total：本次扫描读取的原始数据总行数。
        #     - counts：截至当前的全量两臂人数，依次为对照组、处理组。
        #     - source：原始文件指纹，包含来源路径、大小、修改时间及 SHA256。
        #     - progress：接收扫描进度文本的回调函数。
        #     - include_audit_stats：是否额外整理全量特征范围和组内标准差，供审计复用。
        # 输出参数：
        #     - (全量平衡表, C2ST 样本, SRM 字典, 审计特征统计)；未请求审计统计时末项为 None。
        #-------------
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
            if include_audit_stats:
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
        return balance, c2st_sample, srm, feature_stats if include_audit_stats else None
