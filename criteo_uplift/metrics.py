"""纯排序指标公式与固定评分的配对 Bootstrap；不训练模型或选择策略。"""
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from .input_checks import check_binary_vector

RANKING_METRIC_DESCRIPTION = (
    'v2：横轴为总体覆盖比例，所有面积使用完整排序前缀与原点作梯形积分。'
    'HT 曲线累计固定分配概率加权的结果；Qini 扣除连接原点和终点的随机基线面积。'
    '代入估计使用前缀内两组结果率之差乘覆盖比例，需要两组均存在。'
    '传统 Qini 使用处理组正例数减去按组别人数比例折算的对照组正例数，再除以总体人数；'
    '仅要求前缀存在对照组，保留仅有对照组时的零值点，并扣除自身随机基线面积。'
    '初始未定义区间仅在积分时线性连接，不作完美排序归一化，也不压缩积分点。'
    '历史 v1 使用同时存在两组的前缀；这些历史结果保持原样。'
)



def _binary_inputs(y, t):
    check_binary_vector(y, "y")
    check_binary_vector(t, "t", len(y), require_both=True)
    return np.asarray(y, dtype=float), np.asarray(t, dtype=np.int8)


def _scores(scores, n):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise ValueError("scores must be finite, one-dimensional, and match y")
    return values

def _ht_outcome(y, t, e):
    if not np.isscalar(e) or not np.isfinite(e) or not 0 < e < 1:
        raise ValueError(
            "e must be a fixed assignment probability strictly between 0 and 1"
        )
    return t * y / e - (1 - t) * y / (1 - e)

def _rank(scores, seed):
    # 用独立随机种子处理同分，排序不依赖结果标签。
    # 避免 CSV 中可能按处理组排列的原始行序决定同分策略。
    tie_break = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((tie_break, -scores))

def uplift_curve(y, t, scores, e: float, seed: int = 42) -> pd.DataFrame:
    y, t = _binary_inputs(y, t)
    n = len(y)
    scores = _scores(scores, n)
    order = _rank(scores, seed)
    prefix = _prefix_statistics(y[order], t[order], e)
    fraction = prefix["n_selected"] / n
    uplift = prefix["uplift_rate"]
    lift = uplift / uplift[-1] if uplift[-1] > 1e-12 else np.full(n + 1, np.nan)
    plugin_gain = fraction * uplift
    plugin_gain[0] = 0.0
    result = pd.DataFrame(
        {
            "fraction": fraction,
            "n_selected": prefix["n_selected"],
            "gain_ht": prefix["gain_ht"],
            "random_gain_ht": fraction * prefix["gain_ht"][-1],
            "qini_traditional": prefix["qini_traditional"],
            "uplift": uplift,  # 兼容旧字段名：累计两组结果率之差。
            "cumulative_lift": uplift,
            "uplift_rate": uplift,
            "gain_plugin": plugin_gain,
            "lift": lift,  # 兼容旧字段名：相对累计提升。
            "relative_lift": lift,
            "n_treatment": prefix["n_treatment"],
            "n_control": prefix["n_control"],
            "responses_treatment": prefix["responses_treatment"],
            "responses_control": prefix["responses_control"],
        }
    )
    result.attrs.update({"assignment_probability": float(e), "tie_break_seed": seed})
    return result

def _prefix_statistics(y, t, e):
    """计算已排序样本的累计量，供曲线展示和排序指标共用；包含原点。"""
    n = len(y)
    count = np.arange(n + 1)
    n1 = np.r_[0, np.cumsum(t, dtype=np.int64)]
    n0 = count - n1
    r1 = np.r_[0.0, np.cumsum(t * y)]
    r0 = np.r_[0.0, np.cumsum((1 - t) * y)]
    traditional = np.full(n + 1, np.nan)
    has_control = n0 > 0
    traditional[has_control] = (
        r1[has_control] - r0[has_control] * n1[has_control] / n0[has_control]
    ) / n
    uplift = np.full(n + 1, np.nan)
    has_both = (n1 > 0) & (n0 > 0)
    uplift[has_both] = r1[has_both] / n1[has_both] - r0[has_both] / n0[has_both]
    return {
        "n_selected": count,
        "n_treatment": n1,
        "n_control": n0,
        "responses_treatment": r1,
        "responses_control": r0,
        "gain_ht": np.r_[0.0, np.cumsum(_ht_outcome(y, t, e))] / n,
        "qini_traditional": traditional,
        "uplift_rate": uplift,
    }

def _area(x, values):
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))

def _ranking_from_ordered(y, t, e):
    n = len(y)
    prefix = _prefix_statistics(y, t, e)
    cumulative = prefix["gain_ht"]
    auuc = float(np.sum((cumulative[:-1] + cumulative[1:]) * 0.5) / n)
    ate = float(cumulative[-1])
    count = prefix["n_selected"]
    # 前缀只有对照组时，传统 Qini 值为零，应保留这些已定义的点。
    # 早期 v1 版本只对同时存在两组的前缀进行积分。
    control = prefix["n_control"] > 0
    q_fraction = np.r_[0.0, count[control] / n]
    traditional = np.r_[0.0, prefix["qini_traditional"][control]]
    both = (prefix["n_treatment"] > 0) & control
    rates = prefix["uplift_rate"][both]
    fraction = np.r_[0.0, count[both] / n]
    plugin_gain = np.r_[0.0, count[both] / n * rates]
    plugin_area = _area(fraction, plugin_gain)
    return {
        "n": n,
        "auuc_ht": auuc,
        "qini_ht": auuc - ate / 2,
        "ate_ht": ate,
        "auuc_plugin": plugin_area,
        "auuc_plugin_centered": plugin_area - float(rates[-1]) / 2,
        "qini_score_traditional": _area(q_fraction, traditional)
        - float(traditional[-1]) / 2,
    }

def ranking_metrics(y, t, scores, e: float, seed: int = 42) -> dict:
    y, t = _binary_inputs(y, t)
    scores = _scores(scores, len(y))
    order = _rank(scores, seed)
    return _ranking_from_ordered(y[order], t[order], e)

def ranking_confidence_intervals(
    y,
    t,
    score_dict: Mapping[str, Sequence[float]],
    e: float,
    bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    y, t = _binary_inputs(y, t)
    _ht_outcome(y, t, e)
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero or an integer of at least 2")
    if not score_dict or any(not isinstance(k, str) or not k for k in score_dict):
        raise ValueError("score_dict requires nonempty string policy names")
    names = list(score_dict)
    metrics = [
        "auuc_ht",
        "qini_ht",
        "auuc_plugin",
        "auuc_plugin_centered",
        "qini_score_traditional",
    ]
    n = len(y)
    orders = [_rank(_scores(score_dict[k], n), seed) for k in names]

    def values(order):
        result = _ranking_from_ordered(y[order], t[order], e)
        return [result[k] for k in metrics]

    point = np.asarray([values(order) for order in orders])
    boot = np.empty((bootstrap, len(names), len(metrics)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 271828]))
    arm1, arm0 = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    for b in range(bootstrap):
        draw1 = arm1[rng.integers(len(arm1), size=len(arm1))]
        draw0 = arm0[rng.integers(len(arm0), size=len(arm0))]
        counts = np.bincount(draw1, minlength=n) + np.bincount(draw0, minlength=n)
        for j, order in enumerate(orders):
            boot[b, j] = values(np.repeat(order, counts[order]))
    ci = (
        np.quantile(boot, [0.025, 0.975], axis=0)
        if bootstrap
        else np.full((2, *point.shape), np.nan)
    )
    ref = names.index("propensity") if "propensity" in names else None
    delta = point - point[ref] if ref is not None else np.full_like(point, np.nan)
    delta_ci = (
        np.quantile(boot - boot[:, ref : ref + 1], [0.025, 0.975], axis=0)
        if bootstrap and ref is not None
        else np.full_like(ci, np.nan)
    )
    return pd.DataFrame(
        [
            {
                "policy": name,
                "metric": metric,
                "estimate": point[j, m],
                "ci_low": ci[0, j, m],
                "ci_high": ci[1, j, m],
                "vs_propensity": delta[j, m],
                "vs_propensity_ci_low": delta_ci[0, j, m],
                "vs_propensity_ci_high": delta_ci[1, j, m],
                "bootstrap": bootstrap,
            }
            for j, name in enumerate(names)
            for m, metric in enumerate(metrics)
        ]
    )

def _prefix_values(z_sorted, counts_sorted, k):
    cumulative_n = np.cumsum(counts_sorted)
    cumulative_z = np.cumsum(counts_sorted * z_sorted)
    selected = np.zeros(len(k), dtype=float)
    positive = k > 0
    boundary = np.searchsorted(cumulative_n, k[positive], side="left")
    before_n = np.where(boundary > 0, cumulative_n[np.maximum(boundary - 1, 0)], 0)
    before_z = np.where(boundary > 0, cumulative_z[np.maximum(boundary - 1, 0)], 0.0)
    selected[positive] = before_z + (k[positive] - before_n) * z_sorted[boundary]
    return selected
