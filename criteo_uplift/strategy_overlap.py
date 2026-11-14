"""解释冻结策略的选人差异；只读取保存评分，不训练、不重新选型。"""
from pathlib import Path
import hashlib

import numpy as np
from scipy.stats import spearmanr

from .input_checks import check_row_ids
from .metrics import _scores_check


def score_overlap(row_ids, scores_a, scores_b, *, budget=.2, seed=42):
    """复用 metrics.py 的共享随机优先级、降序 lexsort 和 floor(qN)。"""
    row_ids = np.asarray(row_ids)
    check_row_ids(row_ids, n_rows=len(row_ids))
    a, b = _scores_check(scores_a, len(row_ids)), _scores_check(scores_b, len(row_ids))
    if not np.isscalar(budget) or not np.isfinite(budget) or not 0 < budget <= 1:
        raise ValueError('budget 必须属于 (0, 1]')
    n = len(row_ids)
    k = int(np.floor(budget * n + 1e-12))
    if k == 0:
        raise ValueError('样本不足以选出至少一行')
    priority = np.random.default_rng(seed).random(n)
    order_a = np.lexsort((priority, -a))[:k]
    order_b = np.lexsort((priority, -b))[:k]
    common = int(np.intersect1d(row_ids[order_a], row_ids[order_b], assume_unique=True).size)
    rho = None if np.ptp(a) == 0 or np.ptp(b) == 0 else float(spearmanr(a, b).statistic)
    return {
        'n': n, 'budget': float(budget), 'k': k, 'seed': int(seed),
        'common': common, 'only_a': k - common, 'only_b': k - common,
        'union': 2 * k - common, 'overlap': common / k,
        'jaccard': common / (2 * k - common), 'spearman': rho,
        'common_fraction_of_all': common / n, 'only_a_fraction_of_all': (k - common) / n,
        'only_b_fraction_of_all': (k - common) / n, 'only_a_fraction_of_k': (k - common) / k,
        'only_b_fraction_of_k': (k - common) / k,
        'tie_rule': 'np.lexsort((np.random.default_rng(seed).random(N), -score)); both strategies share priorities',
        'spearman_rule': '原始全部评分的平均并列秩相关；不把破同分优先级加入 Spearman；常数评分时未定义',
        'interpretation': '仅描述冻结评分的策略行为，不使用结果标签，不用于重新选型或证明因果机制。',
    }


def saved_score_overlap(path, policy, reference, *, budget=.2, seed=42):
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        result = score_overlap(saved['row_id'], saved[policy], saved[reference], budget=budget, seed=seed)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {**result, 'policy': policy, 'reference': reference,
            'source_file': 'evaluation/predictions.npz', 'source_sha256': digest.hexdigest()}
