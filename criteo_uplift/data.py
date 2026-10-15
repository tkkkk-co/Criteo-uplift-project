"""只读原始 CSV;全量流式审计与等概率 reservoir 抽样。

每行独立 U(0,1) priority,保留最小 k 个，等价无放回简单随机抽样。
不按 treatment/outcome 过采样，不把文件头 k 行误当随机样本。
"""

# 流式处理（Streaming Processing）：不一次性将所有数据读入内存，而是分批读取和处理。

# Reservoir Sampling（水库抽样）：一种从大量数据中随机保留固定数量样本的方法。
# 不过，这份代码采用的是带随机优先级的水库抽样，而不是经典的逐行替换式水库抽样。


from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [f'f{i}' for i in range(12)]
LABELS = ['treatment', 'conversion', 'visit', 'exposure']
COLUMNS = FEATURES + LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'
DEFAULT_DATA = DATA_ROOT / 'raw/criteo-uplift-v2.1.csv'

# 保护原始数据

def safe_output(path):
    """拒绝向受保护 data/（包括 symlink 指向）写入结果。"""
    path = Path(path).resolve()
    if path == DATA_ROOT.resolve() or DATA_ROOT.resolve() in path.parents:
        raise ValueError('data/ 只读：请把输出放到 artifacts/。')
    return path


def write_json(value, path):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(v):
        if isinstance(v, dict): return {str(k): convert(x) for k,x in v.items()}
        if isinstance(v, (tuple,list)): return [convert(x) for x in v]
        if isinstance(v, np.ndarray): return convert(v.tolist())
        if isinstance(v, np.generic): return convert(v.item())
        if isinstance(v,float) and not np.isfinite(v): return None
        if isinstance(v, Path): return str(v)
        return v
    path.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False))


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()

    with path.open('rb') as f:
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


def scan_and_sample(path=DEFAULT_DATA, sample_size=1_000_000, seed=42, chunk_size=250_000):
    """返回 (sample DataFrame, audit dict)。sample_size=0 保留全量；不写文件。

    重复行仅诊断不删除；没有 user ID，不能把匿名特征相同推断为重复用户。
    重复诊断基于原始 float64 全字段的 64-bit hash，存在极小碰撞可能。
    """
    path = Path(path)
    if sample_size < 0 or chunk_size < 1: 
        raise ValueError('sample_size>=0, chunk_size>=1')
    if list(pd.read_csv(path, nrows=0).columns) != COLUMNS:
        raise ValueError(f'CSV schema 必须恰为 {COLUMNS}')

    rng = np.random.default_rng(seed)
    reservoir = np.empty((0,len(COLUMNS)), dtype=np.float64)
    row_ids = np.empty(0,dtype=np.int64) # 随机抽样的样本在原全量数据的index
    priorities = np.empty(0) #优先级

    hashes, all_chunks = [], []

    counts = np.zeros(2,dtype=np.int64) #计数 C T组样本数
    positives = np.zeros((2,3),dtype=np.int64) #计数 C T组正样本数（CVR visit exposure）
    sums = np.zeros((2,12)); 
    squares = np.zeros((2,12)) #计数 C T组特征和、平方和
    minimum = np.full(12,np.inf); 
    maximum = np.full(12,-np.inf)

    n = 0;  #row_id：全量数据的行号
    bad_visit = 0; #bad_visit：CVR=1但visit=0的样本数；
    control_exposure = 0 # control_exposure：control组中exposure=1的样本数


# 将 CSV 按批次读取成 DataFrame，检查每一行的数据是否合法，并为每一行生成哈希指纹。
    with pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype={
            **{f: 'float64' for f in FEATURES},
            **{l: 'float64' for l in LABELS},
        },
    ) as reader:

        for chunk in reader:
            values = chunk.to_numpy() #values:chunk的numpy数组表示
            if not np.isfinite(values).all():
                raise ValueError(
                    f'row offset {n}: 存在 missing/non-finite，请先明确处理方案。'
                )

            if not np.isin(values[:, 12:], [0, 1]).all():
                raise ValueError('标签必须为0/1')
            hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())

            t = chunk.treatment.to_numpy(dtype=np.int8)
            x = values[:, :12]

            for a in [0, 1]:
                sub = x[t == a]
                counts[a] += len(sub) # T/C 组样本量
                sums[a] += sub.sum(axis=0)  #sums[a, j]  第 a 组、第 j 个特征的总和
                squares[a] += np.square(sub).sum(axis=0) # squares[a, j]  第 a 组、第 j 个特征的平方和
                positives[a] += values[t == a, 13:].sum(axis=0).astype(np.int64)

            minimum = np.minimum(minimum, x.min(axis=0)) 

            maximum = np.maximum(maximum, x.max(axis=0))

            bad_visit += int(((chunk.conversion == 1) & (chunk.visit == 0)).sum())

            control_exposure += int(
                ((chunk.treatment == 0) & (chunk.exposure == 1)).sum()
            )

            # 对每个chunk迭代编号，确保每个样本都有唯一的 row_id。 左闭右开 编号从0开始
            ids = np.arange(n, n + len(chunk), dtype=np.int64) 

            # 如果 sample_size=0，则保留所有数据；否则使用 reservoir sampling 抽样。
            if sample_size == 0:
                chunk = chunk.astype(
                    {
                        **{f: 'float32' for f in FEATURES},
                        **{l: 'int8' for l in LABELS},
                    }
                )
                chunk.insert(0, 'row_id', ids)
                all_chunks.append(chunk)
            # 随机优先级抽样 -给每一行分配一个独立的 U(0,1) priority，保留最小 k 个，等价于无放回简单随机抽样。
            else:
                keys = rng.random(len(chunk))
                reservoir = np.concatenate([reservoir, values])
                priorities = np.concatenate([priorities, keys])
                row_ids = np.concatenate([row_ids, ids]) #追踪样本来源
                #如果候选样本超过 k，保留最小的 k 个
                if len(reservoir) > sample_size:
                    keep = np.argpartition(priorities, sample_size - 1)[:sample_size]
                    reservoir = reservoir[keep]
                    priorities = priorities[keep]
                    row_ids = row_ids[keep]
            n += len(chunk)
            if n % 1_000_000 == 0:
                print(f'  scanned {n:,} rows', flush=True)

    if n == 0 or (counts == 0).any():
        raise ValueError('数据为空或缺少treatment/control')

    unique_hashes = np.unique(np.concatenate(hashes)).size
    
    if sample_size == 0:
        sample = pd.concat(all_chunks, ignore_index=True)
    else:
        order = np.argsort(row_ids)
        sample = pd.DataFrame(reservoir[order], columns=COLUMNS)
        sample = sample.astype(
            {
                **{f: 'float32' for f in FEATURES},
                **{l: 'int8' for l in LABELS},
            }
        )
        sample.insert(0, 'row_id', row_ids[order])

    feature_stats = []
    for j, f in enumerate(FEATURES):
        row = {
            'feature': f,
            'min': minimum[j],
            'max': maximum[j],
        }
        for a in [0, 1]:
            row[f'mean_{a}'] = sums[a, j] / counts[a]
            row[f'std_{a}'] = (
                np.sqrt(
                    max(
                        0,
                        (squares[a, j] - sums[a, j] ** 2 / counts[a]) / (counts[a] - 1),
                    )
                )
                if counts[a] > 1
                else np.nan
            )
        denominator = np.sqrt((row['std_0'] ** 2 + row['std_1'] ** 2) / 2)
        difference = row['mean_1'] - row['mean_0']
        row['smd'] = (
            difference / denominator
            if denominator > 0
            else (0.0 if denominator == 0 and difference == 0 else np.nan)
        )
        row['smd_status'] = (
            'computed'
            if np.isfinite(row['smd'])
            else 'undefined_zero_or_missing_variance'
        )
        feature_stats.append(row)

    audit = {
        'n_rows': n,
        'n_columns': 16,
        'n_treatment': counts[1],
        'n_control': counts[0],
        'treatment_ratio': counts[1] / n,
        'missing_or_nonfinite': 0,
        'invalid_binary_labels': 0,
        'control_exposed': control_exposure,
        'conversion_without_visit': bad_visit,
        'duplicate_row_fingerprints': n - int(unique_hashes),
        'duplicate_method': '64-bit hash of all original float64 fields; diagnostic only; no user ID; not deduplicated',
        'feature_stats': feature_stats,
        'sample_n': len(sample),
        'sample_seed': seed,
        'sampling': (
            'all rows'
            if sample_size == 0
            else 'uniform without replacement, random-priority reservoir'
        ),
        'groups': {
            str(a): {
                'n': counts[a],
                **{
                    label: {
                        'positive': positives[a, k],
                        'rate': positives[a, k] / counts[a],
                    }
                    for k, label in enumerate(LABELS[1:])
                },
            }
            for a in [0, 1]
        },
    }
    return sample, audit


def save_sample(sample, path):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row_id=sample.row_id.to_numpy(),
        X=sample[FEATURES].to_numpy(dtype=np.float32),
        labels=sample[LABELS].to_numpy(dtype=np.int8),
    )


def load_sample(path):
    with np.load(path, allow_pickle=False) as z:
        frame = pd.DataFrame(z['X'], columns=FEATURES)
        for j, c in enumerate(LABELS):
            frame[c] = z['labels'][:, j]
        frame.insert(0, 'row_id', z['row_id'])
    return frame


def split_sample(df, outcome='conversion', seed=42):
    """60/20/20 分层 T×Y；所有模型选择仅可用 validation。"""
    if outcome not in ['conversion', 'visit']:
        raise ValueError('unknown outcome')
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
    return {
        name: df.iloc[pos].reset_index(drop=True)
        for name, pos in [('train', train), ('validation', val), ('test', test)]
    }
