#-------------
# 纯排序指标公式与固定评分的配对 Bootstrap；不训练模型或选择策略。
#-------------
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
    #-------------
    # 函数名：_binary_inputs
    # 功能：统一校验结果变量和处理分配变量，并转换为指标计算所需的 NumPy 类型。
    # 输入参数：
    #     - y：二元结果变量向量。
    #     - t：二元处理分配向量，要求两组均存在。
    # 输出参数：
    #     - 二元数组二元组：(float 类型的 y, int8 类型的 t)。
    #-------------
    check_binary_vector(y, "y")
    check_binary_vector(t, "t", len(y), require_both=True)
    return np.asarray(y, dtype=float), np.asarray(t, dtype=np.int8)


def _scores(scores, n):
    #-------------
    # 函数名：_scores
    # 功能：校验策略分数为有限的一维向量，且行数与样本数量一致。
    # 输入参数：
    #     - scores：待检查的策略评分向量。
    #     - n：预期样本行数。
    # 输出参数：
    #     - 校验后的 float NumPy 一维评分数组。
    #-------------
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise ValueError("scores 必须为有限值一维向量，且长度与 y 一致")
    return values

def _ht_outcome(y, t, e):
    #-------------
    # 函数名：_ht_outcome
    # 功能：按固定处理分配概率计算每行的 Horvitz–Thompson 加权结果。
    # 输入参数：
    #     - y：二元结果变量向量。
    #     - t：二元处理分配向量。
    #     - e：固定 Treatment 分配概率，必须严格位于 0 和 1 之间。
    # 输出参数：
    #     - 逐行 HT 结果贡献的一维 NumPy 数组。
    #-------------
    if not np.isscalar(e) or not np.isfinite(e) or not 0 < e < 1:
        raise ValueError(
            "e 必须为严格介于 0 和 1 之间的固定分配概率"
        )
    return t * y / e - (1 - t) * y / (1 - e)

def _rank(scores, seed):
    #-------------
    # 函数名：_rank
    # 功能：按策略分数从高到低生成排序索引，并使用独立随机流稳定处理同分。
    # 输入参数：
    #     - scores：每个样本的策略分数。
    #     - seed：同分随机打破的种子。
    # 输出参数：
    #     - 从高分到低分排列的样本位置索引数组。
    #-------------
    # 用独立随机种子处理同分，排序不依赖结果标签。
    # 避免 CSV 中可能按处理组排列的原始行序决定同分策略。
    tie_break = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((tie_break, -scores))

def uplift_curve(y, t, scores, e: float, seed: int = 42) -> pd.DataFrame:
    #-------------
    # 函数名：uplift_curve
    # 功能：根据策略分数排序，计算每个覆盖前缀的 HT 增量、传统 Qini 和累计 uplift 曲线。
    # 输入参数：
    #     - y：二元结果变量。
    #     - t：二元处理分配变量。
    #     - scores：每个样本的策略分数。
    #     - e：固定处理分配概率。
    #     - seed：同分排序随机种子。
    # 输出参数：
    #     - Uplift Curve DataFrame，包含覆盖比例、累计人数、HT gain、Qini 和两组累计统计量。
    #-------------
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
    #-------------
    # 函数名：_prefix_statistics
    # 功能：计算已排序样本从原点到每个前缀的累计人数、正例数、HT gain、传统 Qini 和 uplift。
    # 输入参数：
    #     - y：已按策略顺序排列的二元结果向量。
    #     - t：与 y 对齐的处理分配向量。
    #     - e：固定处理分配概率。
    # 输出参数：
    #     - 包含各前缀累计统计数组的字典。
    #-------------
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
    #-------------
    # 函数名：_area
    # 功能：使用梯形积分计算给定横坐标和曲线值对应的面积。
    # 输入参数：
    #     - x：按顺序排列的横坐标数组。
    #     - values：与 x 等长的曲线纵坐标数组。
    # 输出参数：
    #     - 曲线下面积的浮点数。
    #-------------
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))

def _ranking_from_ordered(y, t, e):
    #-------------
    # 函数名：_ranking_from_ordered
    # 功能：对已经按策略分数排序的样本计算 HT AUUC、HT Qini、Plug-in AUUC 和传统 Qini。
    # 输入参数：
    #     - y：已经按策略排序的二元结果向量。
    #     - t：与 y 对齐的二元处理分配向量。
    #     - e：固定处理分配概率。
    # 输出参数：
    #     - 排序指标字典，包含样本量和各类 AUUC/Qini 指标。
    #-------------
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
    #-------------
    # 函数名：ranking_metrics
    # 功能：对单个策略分数排序，并计算该策略的完整排序指标。
    # 输入参数：
    #     - y：二元结果变量。
    #     - t：二元处理分配变量。
    #     - scores：策略评分向量。
    #     - e：固定处理分配概率。
    #     - seed：同分排序随机种子。
    # 输出参数：
    #     - 排序指标字典，包含 HT AUUC、Qini、Plug-in 和传统 Qini 等指标。
    #-------------
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
    #-------------
    # 函数名：ranking_confidence_intervals
    # 功能：对多个固定策略分数执行处理臂内配对 Bootstrap，计算排序指标区间和相对倾向基线的配对区间。
    # 输入参数：
    #     - y：二元结果变量。
    #     - t：二元处理分配变量。
    #     - score_dict：策略名称到评分向量的映射。
    #     - e：固定处理分配概率。
    #     - bootstrap：Bootstrap 重抽样次数，允许 0 或大于等于 2 的整数。
    #     - seed：排序和 Bootstrap 随机种子。
    # 输出参数：
    #     - 长表 DataFrame：每个策略和指标的点估计、95% CI 及相对 propensity 的配对差值区间。
    #-------------
    y, t = _binary_inputs(y, t)
    _ht_outcome(y, t, e)
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap 必须为 0 或大于等于 2 的整数")
    if not score_dict or any(not isinstance(k, str) or not k for k in score_dict):
        raise ValueError("score_dict 的策略名称必须为非空字符串")
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
        #-------------
        # 函数名：ranking_confidence_intervals.values
        # 功能：根据一个排序索引计算该策略的完整排序指标向量，供 Bootstrap 点估计和重抽样共用。
        # 输入参数：
        #     - order：当前重排样本的位置索引。
        # 输出参数：
        #     - 按预设指标顺序排列的指标值列表。
        #-------------
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
    #-------------
    # 函数名：_prefix_values
    # 功能：在压缩后的分数分组表示中，计算指定 top-k 前缀对应的加权累计结果。
    # 输入参数：
    #     - z_sorted：按分数排序的单行结果贡献。
    #     - counts_sorted：每个分数位置对应的重抽样权重/出现次数。
    #     - k：需要计算的前缀样本数量数组。
    # 输出参数：
    #     - 每个 k 对应的累计结果值数组。
    #-------------
    cumulative_n = np.cumsum(counts_sorted)
    cumulative_z = np.cumsum(counts_sorted * z_sorted)
    selected = np.zeros(len(k), dtype=float)
    positive = k > 0
    boundary = np.searchsorted(cumulative_n, k[positive], side="left")
    before_n = np.where(boundary > 0, cumulative_n[np.maximum(boundary - 1, 0)], 0)
    before_z = np.where(boundary > 0, cumulative_z[np.maximum(boundary - 1, 0)], 0.0)
    selected[positive] = before_z + (k[positive] - before_n) * z_sorted[boundary]
    return selected
