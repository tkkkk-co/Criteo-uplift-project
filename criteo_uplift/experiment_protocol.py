"""Freeze an outcome-stratified holdout before development; source IDs are immutable."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .pipeline_io import safe_output, write_result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_protocol(data, output, *, history, seed=42):
    """Only labels are read for predeclared stratification, never strategy scores."""
    output = safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('协议目录非空，拒绝覆盖已冻结 Test')
    data = Path(data).resolve()
    before = data.stat()
    labels = pd.read_csv(data, usecols=['treatment', 'conversion'], dtype=np.int8)
    if not labels.isin([0, 1]).all().all():
        raise ValueError('处理和转化标签必须为二元值')
    n = len(labels)
    excluded = np.zeros(n, dtype=bool)
    provenance = []
    history = Path(history)
    pattern = 'data_preparation/*_row_ids.csv' if (history / 'data_preparation').is_dir() else '*/data_preparation/*_row_ids.csv'
    for path in sorted(history.glob(pattern)):
        ids = pd.read_csv(path).row_id.to_numpy(dtype=np.int64)
        if ((ids < 0) | (ids >= n)).any():
            raise ValueError(f'历史行号超出源文件范围：{path}')
        excluded[ids] = True
        provenance.append({'path': str(path.resolve()), 'n': len(ids), 'sha256': sha256_file(path)})
    pool = np.flatnonzero(~excluded)
    test_n = int(np.ceil(n * .2))
    if len(pool) <= test_n:
        raise ValueError('排除历史记录后的样本池不足以冻结 20% Test')
    strata = (labels.treatment * 2 + labels.conversion).to_numpy()
    _, test = train_test_split(pool, test_size=test_n, stratify=strata[pool], random_state=seed)
    test = np.sort(test)
    source_hash = sha256_file(data)
    after = data.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('冻结期间源文件发生变化')
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'test_row_ids.npy', test)
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
        'source_sha256': source_hash, 'source_n': n, 'test_n': len(test),
        'test_ids_sha256': sha256_file(output / 'test_row_ids.npy'),
        'history_files': provenance, 'historical_rows_excluded': int(excluded.sum()),
        'primary_metric': 'gain_ht', 'primary_budget': .2, 'bootstrap': 200, 'trees': 150,
        'c2st_sample': 20000, 'permutations': 499,
        'test_selection': 'T × conversion stratified random holdout from rows outside all traceable historical modeling splits',
        'limitations': '历史全量审计、ATE 和特征诊断已被查看。仅能核验保存的历史行号，无法证明未留存的数据使用情况；匿名特征不能验证用户级独立性。最终 Test 对本轮新增模型开发保持行级隔离，但不是从未接触过的新实验。',
    }
    write_result(protocol, output / 'protocol.json')
    return protocol


def load_protocol(path):
    path = Path(path)
    metadata = json.loads((path / 'protocol.json').read_text())
    if sha256_file(path / 'test_row_ids.npy') != metadata['test_ids_sha256']:
        raise ValueError('冻结 Test 行号指纹不匹配')
    test = np.load(path / 'test_row_ids.npy', allow_pickle=False)
    if len(test) != metadata['test_n'] or len(np.unique(test)) != len(test):
        raise ValueError('冻结 Test 行号无效')
    return metadata, test


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history', type=Path, default=Path('artifacts'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(freeze_protocol(args.data, args.output, history=args.history, seed=args.seed), ensure_ascii=False, indent=2))
