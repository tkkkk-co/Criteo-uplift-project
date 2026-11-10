# 评价随机实验的平均效果、增量排序和触达策略；模型分数必须来自未使用评价结果进行拟合的模型。
# Notebook 02–05 使用本模块；这里尽量只计算指标，不负责训练、CLI、产物目录或 HTML。
# 指标产物与 Notebook 对照见 docs/RESULTS_GUIDE.md。
# 分配概率固定使用外部值或训练集估计值；置信区间不包含重新训练和选模的不确定性。

from __future__ import annotations

from math import sqrt
from scipy.special import log_ndtr
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = (0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
_Z_975 = 1.959963984540054

# 将指标口径显式保存到运行记录，避免依赖函数注释的存储形式。
RANKING_METRIC_DESCRIPTION = (
    'v2：横轴为总体覆盖比例，所有面积使用完整排序前缀与原点作梯形积分。'
    'HT 曲线累计固定分配概率加权的结果；Qini 扣除连接原点和终点的随机基线面积。'
    '代入估计使用前缀内两组结果率之差乘覆盖比例，需要两组均存在。'
    '传统 Qini 使用处理组正例数减去按组别人数比例折算的对照组正例数，再除以总体人数；'
    '仅要求前缀存在对照组，保留仅有对照组时的零值点，并扣除自身随机基线面积。'
    '初始未定义区间仅在积分时线性连接，不作完美排序归一化，也不压缩积分点。'
    '历史 v1 使用同时存在两组的前缀；这些历史结果保持原样。'
)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_binary_inputs —— 检查结果标签和处理标签。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
# 输出：
#     y：校验后的结果标签数组。
#     t：校验后的处理标签数组。
# 说明：确认两个标签数组非空、等长且只包含 0 和 1，并要求两组均有记录。
#-----------------------------------------------------------------------------------------------------------------------
def _binary_inputs(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.shape != t.shape or y.size == 0:
        raise ValueError(
            "y and t must be nonempty one-dimensional arrays of equal length"
        )
    if not np.isin(y, (0, 1)).all() or not np.isin(t, (0, 1)).all():
        raise ValueError("y and t must contain only finite binary values 0 and 1")
    if not np.any(t == 0) or not np.any(t == 1):
        raise ValueError("evaluation requires both treatment arms")
    return y, t.astype(np.int8)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_scores —— 检查预测分数的长度和取值。
# 输入：
#     scores：与样本顺序一致的预测分数。
#     n：预期的预测分数个数。
# 输出：
#     values：有限且与样本数量一致的分数数组。
# 说明：把分数转换为一维数值数组，并检查长度及所有取值是否有限。
#-----------------------------------------------------------------------------------------------------------------------
def _scores(scores, n):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise ValueError("scores must be finite, one-dimensional, and match y")
    return values


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_ht_outcome —— 按固定分配概率转换逐行结果。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     e：固定的 Treatment 分配概率，通常来自训练集。
# 输出：
#     加权结果：每条记录对增量效果估计的贡献数组。
# 说明：按固定处理分配概率对两组结果加权，为总体增量和策略收益计算提供逐行贡献。
#-----------------------------------------------------------------------------------------------------------------------
def _ht_outcome(y, t, e):
    if not np.isscalar(e) or not np.isfinite(e) or not 0 < e < 1:
        raise ValueError(
            "e must be a fixed assignment probability strictly between 0 and 1"
        )
    return t * y / e - (1 - t) * y / (1 - e)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_rank —— 按分数排序并独立处理同分记录。
# 输入：
#     scores：与样本顺序一致的预测分数。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     order：从高分到低分的样本位置索引。
# 说明：先按分数从高到低排序；分数相同时使用独立随机优先级，避免依赖 CSV 行序。
#-----------------------------------------------------------------------------------------------------------------------
def _rank(scores, seed):
    # 用独立随机种子处理同分，排序不依赖结果标签。
    # 避免 CSV 中可能按处理组排列的原始行序决定同分策略。
    tie_break = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((tie_break, -scores))


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_normal_pvalue —— 稳定计算双侧标准正态尾概率及其展示文本。
# 输入：z：原有检验公式得到的 z 统计量，可为未定义值。
# 输出：p_value、log_p_value（自然对数）、p_value_text（科学计数法，约四位有效数字）。
# 说明：使用 log(2)+log(Phi(-abs(z)))；不设置概率下限。浮点概率下溢时，
#       仍保存有限对数概率，并从它生成科学计数文本；文本不是人为截断的上界。
#-----------------------------------------------------------------------------------------------------------------------
def _normal_pvalue(z):
    log_p = float(np.log(2.0) + log_ndtr(-abs(z)))
    p = float(np.exp(log_p))
    if np.isnan(log_p):
        text = None
    elif np.isfinite(log_p) and p == 0:
        exponent = int(np.floor(log_p / np.log(10.0)))
        mantissa = np.exp(log_p - exponent * np.log(10.0))
        if round(float(mantissa), 3) >= 10:
            mantissa /= 10
            exponent += 1
        text = f"{mantissa:.3f}e{exponent:+d}"
    else:
        text = f"{p:.3e}" if p > 0 else "0 (non-finite test statistic)"
    return p, log_p, text


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_mean_difference —— 计算两组平均结果的差异和不确定性。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
# 输出：
#     统计结果：包含组别人数、结果率、平均处理效应、标准误、区间和 p 值的字典。
# 说明：比较两组结果率；区间反映组内结果变动，不能据此验证因果识别假设。
#-----------------------------------------------------------------------------------------------------------------------
def _mean_difference(y, t):
    treated = y[t == 1]
    control = y[t == 0]
    n1, n0 = len(treated), len(control)
    p1 = float(treated.mean()) if n1 else np.nan
    p0 = float(control.mean()) if n0 else np.nan
    ate = p1 - p0
    # 用两组各自的样本方差计算均值差的标准误。
    # 某组只有一条记录时，无法估计该组方差。
    se = (
        sqrt(float(treated.var(ddof=1) / n1 + control.var(ddof=1) / n0))
        if min(n1, n0) > 1
        else np.nan
    )
    pooled_p = (float(treated.sum()) + float(control.sum())) / (n1 + n0)
    null_se = (
        sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n0)) if min(n1, n0) else np.nan
    )
    if null_se > 0:
        p_value, log_p_value, p_value_text = _normal_pvalue(ate / null_se)
    elif null_se == 0 and ate == 0:
        p_value, log_p_value, p_value_text = _normal_pvalue(0.0)
    else:
        p_value, log_p_value, p_value_text = _normal_pvalue(np.nan)
    return {
        "n": n1 + n0,
        "n_treatment": n1,
        "n_control": n0,
        "response_treatment": p1,
        "response_control": p0,
        "ate": ate,
        "standard_error": se,
        "ci_low": ate - _Z_975 * se,
        "ci_high": ate + _Z_975 * se,
        "pooled_null_standard_error": null_se,
        "p_value": p_value,
        "log_p_value": log_p_value,
        "p_value_text": p_value_text,
        "relative_uplift": ate / p0 if p0 > 0 else np.nan,
    }


#-----------------------------------------------------------------------------------------------------------------------
# 函数：aggregate_ate_from_audit —— 计算两组平均结果的差异和不确定性。
# 输入：
#     audit：全量审计的聚合统计字典。
#     outcome：当前结果列名，通常为 conversion 或 visit。
# 输出：
#     统计结果：包含组别人数、结果率、平均处理效应、标准误、区间和 p 值的字典。
# 说明：比较两组结果率；区间反映组内结果变动，不能据此验证因果识别假设。
#-----------------------------------------------------------------------------------------------------------------------
def aggregate_ate_from_audit(audit, outcome):
    n0, n1 = (
        audit["groups"][str(arm)]["n"]
        for arm in (0, 1)
    )
    r0, r1 = (
        audit["groups"][str(arm)][outcome]["positive"]
        for arm in (0, 1)
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (n0, n1, r0, r1)
    ) or not (n0 > 0 and n1 > 0 and 0 <= r0 <= n0 and 0 <= r1 <= n1):
        raise ValueError('Audit counts must be valid integers with both arms present')
    
    p0, p1 = r0 / n0, r1 / n1
    delta = p1 - p0
    # 沿用已发布全量报告的二项 plug-in 方差；样本 ATE / 校准另用 ddof=1。
    se = np.sqrt(
        p1 * (1 - p1) / n1 
        + p0 * (1 - p0) / n0 
    ) if min(n0, n1) > 1 else np.nan

    pooled = (r0 + r1) / (n0 + n1)
    nullse = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    p_value, log_p_value, p_value_text = _normal_pvalue(delta / nullse if nullse > 0 else 0.0)
    return {
        "outcome": outcome,
        "n": n0 + n1,
        "n_control": n0,
        "n_treatment": n1,
        "response_control": p0,
        "response_treatment": p1,
        "ate": delta,
        "standard_error": se,
        "ci_low": delta - 1.95996398454 * se,
        "ci_high": delta + 1.95996398454 * se,
        "p_value": p_value,
        "log_p_value": log_p_value,
        "p_value_text": p_value_text,
        "relative_uplift": delta / p0 if p0 else None,
    }


#-----------------------------------------------------------------------------------------------------------------------
# 函数：ate_summary —— 计算两组平均结果的差异和不确定性。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
# 输出：
#     统计结果：包含组别人数、结果率、平均处理效应、标准误、区间和 p 值的字典。
# 说明：比较两组结果率；区间反映组内结果变动，不能据此验证因果识别假设。
#-----------------------------------------------------------------------------------------------------------------------
def ate_summary(y, t) -> dict:
    y, t = _binary_inputs(y, t)
    return _mean_difference(y, t)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：uplift_curve —— 生成完整排序前缀上的增量曲线。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     scores：与样本顺序一致的预测分数。
#     e：固定的 Treatment 分配概率，通常来自训练集。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     result：包含覆盖比例、组别人数、HT 增量、传统 Qini 和结果率差异的曲线表。
# 说明：保留所有排序位置，区分不同口径的曲线；缺少必要组别时保留未定义值。
#-----------------------------------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_area —— 计算曲线的梯形面积。
# 输入：
#     x：按顺序排列的横坐标数组。
#     values：需要检查或处理的数值数组。
# 输出：
#     面积：相邻横坐标之间累积得到的浮点数。
# 说明：计算曲线的梯形面积。
#-----------------------------------------------------------------------------------------------------------------------
def _area(x, values):
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_ranking_from_ordered —— 计算完整分辨率的增量排序指标。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     e：固定的 Treatment 分配概率，通常来自训练集。
# 输出：
#     指标：包含样本数、HT 平均效应、各类 AUUC 与 Qini 的字典。
# 说明：区分 HT、代入估计和传统 Qini，不做完美排序归一化；传统 Qini 保留仅有对照组的零值前缀。
#-----------------------------------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：ranking_metrics —— 计算完整分辨率的增量排序指标。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     scores：与样本顺序一致的预测分数。
#     e：固定的 Treatment 分配概率，通常来自训练集。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     指标：包含样本数、HT 平均效应、各类 AUUC 与 Qini 的字典。
# 说明：区分 HT、代入估计和传统 Qini，不做完美排序归一化；传统 Qini 保留仅有对照组的零值前缀。
#-----------------------------------------------------------------------------------------------------------------------
def ranking_metrics(y, t, scores, e: float, seed: int = 42) -> dict:
    y, t = _binary_inputs(y, t)
    scores = _scores(scores, len(y))
    order = _rank(scores, seed)
    return _ranking_from_ordered(y[order], t[order], e)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：ranking_confidence_intervals —— 对冻结排序分数计算配对区间。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     score_dict：策略名称到逐行预测分数数组的映射。
#     e：固定的 Treatment 分配概率，通常来自训练集。
#     bootstrap：重抽样次数；支持时可用 0 仅计算点估计。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     结果表：每个策略、每个指标的点估计、区间及相对响应概率基线的差异。
# 说明：在两组内重抽样，各策略共用同一批记录；区间不包含模型训练和选择的不确定性。
#-----------------------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------------------
    # 函数：values —— 提取指定排序下的各项指标。
    # 输入：
    #     order：样本按分数排列的位置索引。
    # 输出：
    #     指标列表：按外层 metrics 顺序返回的指标数值。
    # 说明：提取指定排序下的各项指标。
    #-----------------------------------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_prefix_values —— 按重抽样次数计算预算内的累计结果。
# 输入：
#     z_sorted：已按策略顺序排列的逐行加权结果。
#     counts_sorted：排序后每条记录在重抽样中的出现次数。
#     k：每个预算对应的选中行数。
# 输出：
#     selected：每个预算对应的加权结果总和数组。
# 说明：按每条记录在重抽样中的出现次数累计结果，并处理预算边界只取部分重复记录的情况。
#-----------------------------------------------------------------------------------------------------------------------
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：evaluate_policies —— 比较各触达策略的增量收益与配对区间。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     score_dict：策略名称到逐行预测分数数组的映射。
#     e：固定的 Treatment 分配概率，通常来自训练集。
#     budgets：需要评价的一组触达比例。
#     bootstrap：重抽样次数；支持时可用 0 仅计算点估计。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     result：各策略、各预算的选中人数、增量结果和相对基线差异表。
# 说明：使用固定分数重选预算内记录，并加入解析随机基线；指标分母是全部符合条件人群。
#-----------------------------------------------------------------------------------------------------------------------
def evaluate_policies(
    y,
    t,
    score_dict: Mapping[str, Sequence[float]],
    e: float,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    y, t = _binary_inputs(y, t)
    n = len(y)
    z = _ht_outcome(y, t, e)
    budget = np.asarray(budgets, dtype=float)
    if (
        budget.ndim != 1
        or not len(budget)
        or not np.isfinite(budget).all()
        or np.any((budget < 0) | (budget > 1))
    ):
        raise ValueError(
            "budgets must be a nonempty sequence of fractions between 0 and 1"
        )
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero or an integer of at least 2")
    if "random" in score_dict:
        raise ValueError("'random' is reserved for the analytic random-policy baseline")
    if any(not isinstance(name, str) or not name for name in score_dict):
        raise ValueError("policy names must be nonempty strings")
    k = np.floor(budget * n + 1e-12).astype(np.int64)
    fractions = k / n
    names = ["random", *score_dict]
    orders = [_rank(_scores(score_dict[name], n), seed) for name in names[1:]]
    sorted_z = [z[order] for order in orders]
    point = np.empty((len(names), len(budget)))
    point[0] = fractions * z.mean()
    for j, values in enumerate(sorted_z, start=1):
        cumulative = np.r_[0.0, np.cumsum(values)]
        point[j] = cumulative[k] / n
    # 全量触达时所有策略使用同一个汇总结果。
    # 避免累加顺序产生微小浮点差异，被误读为策略收益。
    point[:, k == n] = point[0, k == n]

    boot = np.empty((bootstrap, len(names), len(budget)))
    if bootstrap:
        # 重抽样随机流独立于同分排序的随机流。
        rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
        arm1 = np.flatnonzero(t == 1)
        arm0 = np.flatnonzero(t == 0)
        for b in range(bootstrap):
            draws1 = arm1[rng.integers(len(arm1), size=len(arm1))]
            draws0 = arm0[rng.integers(len(arm0), size=len(arm0))]
            counts = np.bincount(draws1, minlength=n) + np.bincount(draws0, minlength=n)
            boot[b, 0] = fractions * np.dot(counts, z) / n
            for j, (order, values) in enumerate(zip(orders, sorted_z), start=1):
                boot[b, j] = _prefix_values(values, counts[order], k) / n
            boot[b, :, k == n] = boot[b, 0, k == n, None]

    scale = 100_000.0
    point *= scale
    boot *= scale
    nan_intervals = np.full((2, len(names), len(budget)), np.nan)
    ci = np.quantile(boot, [0.025, 0.975], axis=0) if bootstrap else nan_intervals
    random_delta = point - point[0]
    random_ci = (
        np.quantile(boot - boot[:, :1], [0.025, 0.975], axis=0)
        if bootstrap
        else nan_intervals
    )
    response_index = names.index("propensity") if "propensity" in names else None
    propensity_delta = (
        point - point[response_index]
        if response_index is not None
        else np.full_like(point, np.nan)
    )
    propensity_ci = (
        np.quantile(
            boot - boot[:, response_index : response_index + 1], [0.025, 0.975], axis=0
        )
        if bootstrap and response_index is not None
        else nan_intervals
    )
    records = []
    for j, name in enumerate(names):
        for b, requested in enumerate(budget):
            records.append(
                {
                    "policy": name,
                    "budget": float(requested),
                    "n_selected": int(k[b]),
                    "selected_fraction": float(fractions[b]),
                    "incremental_per_100k": float(point[j, b]),
                    "ci_low": float(ci[0, j, b]),
                    "ci_high": float(ci[1, j, b]),
                    "vs_random_per_100k": float(random_delta[j, b]),
                    "vs_random_ci_low": float(random_ci[0, j, b]),
                    "vs_random_ci_high": float(random_ci[1, j, b]),
                    "vs_propensity_per_100k": float(propensity_delta[j, b]),
                    "vs_propensity_ci_low": float(propensity_ci[0, j, b]),
                    "vs_propensity_ci_high": float(propensity_ci[1, j, b]),
                    "bootstrap_replicates": int(bootstrap),
                }
            )
    result = pd.DataFrame.from_records(records)
    result.attrs.update(
        {
            "assignment_probability": float(e),
            "ci_method": "paired arm-stratified percentile bootstrap; fixed learned scores; top-k reselected",
            "random_policy": "analytic expected value at the realized selected fraction",
            "seed": seed,
        }
    )
    return result


#-----------------------------------------------------------------------------------------------------------------------
# 函数：calibration_table —— 比较分数分箱中的预测效果与观察效果。
# 输入：
#     y：每行的二分类结果标签。
#     t：每行的处理组标签，0 为 Control，1 为 Treatment。
#     cate：逐行预测的条件平均处理效应。
#     n_bins：分组校准或分箱统计使用的组数。
# 输出：
#     结果表：每箱的人数、预测效应范围、观察效应及描述性区间。
# 说明：分箱只根据预测分数，不用测试结果重新校准模型；缺少组别时观察效应未定义。
#-----------------------------------------------------------------------------------------------------------------------
def calibration_table(y, t, cate, n_bins: int = 10) -> pd.DataFrame:
    y, t = _binary_inputs(y, t)
    cate = _scores(cate, len(y))
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    order = _rank(-cate, seed=42)
    records = []
    for number, selected in enumerate(
        np.array_split(order, min(n_bins, len(y))), start=1
    ):
        summary = _mean_difference(y[selected], t[selected])
        records.append(
            {
                "bin": number,
                "n": len(selected),
                "n_treatment": summary["n_treatment"],
                "n_control": summary["n_control"],
                "predicted_cate": float(cate[selected].mean()),
                "prediction_min": float(cate[selected].min()),
                "prediction_max": float(cate[selected].max()),
                "observed_ate": summary["ate"],
                "standard_error": summary["standard_error"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
            }
        )
    return pd.DataFrame.from_records(records)
