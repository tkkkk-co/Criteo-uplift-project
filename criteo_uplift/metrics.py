#-------------
# 纯排序指标公式与固定评分的配对 Bootstrap；不训练模型或选择策略。
#-------------
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from .input_checks import check_binary_vector

RANKING_METRIC_DESCRIPTION = (
    "按模型预测分数降序排列用户，横轴 q=k/N 表示 Top-k 覆盖比例。"
    "计算每个 Top-k 的累计人数、转化数、Lift 及三种 Gain。"

    "Uplift Gain：使用 Top-k 实际分组比例，估计处理全部 k 人时的总体增量转化率。"
    "Traditional Qini：估计 Top-k 内实际 Treatment 的 N_T(k) 人对应的总体增量转化率。"
    "HT Gain：使用固定分配概率 e 进行逆概率加权，估计处理全部 k 人时的总体增量转化率。"

    "AUUC：对 Gain Curve 使用完整排序前缀进行梯形积分。"
    "Centered AUUC、Traditional Qini Score 和 HT Qini："
    "分别从对应曲线面积中减去终点值的一半。"

    "边界处理：缺少必要实验组的前缀保留 NaN；"
    "Traditional Qini 在仅有 Control 的前缀定义为 0。"
    "积分从原点连接至首个有效节点，不进行完美排序归一化。"
)


def _binary_inputs_check(y, t):
    # 校验二元结果和两臂分配，返回 float 结果与 int8 分配数组。
    check_binary_vector(y, "y")
    check_binary_vector(t, "t", len(y), require_both=True)
    return np.asarray(y, dtype=float), np.asarray(t, dtype=np.int8)


def _scores_check(scores, n):
    # 评分必须是一维有限值向量，且与评价样本逐行对齐。
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise ValueError("scores 必须为有限值一维向量，且长度与 y 一致")
    return values


def _ht_outcome(y, t, e):
    # Z_i = T_i Y_i/e - (1-T_i)Y_i/(1-e)，使用固定分配概率。
    if not np.isscalar(e) or not np.isfinite(e) or not 0 < e < 1:
        raise ValueError(
            "e 必须为严格介于 0 和 1 之间的固定分配概率"
        )
    return t * y / e - (1 - t) * y / (1 - e)




def _gain_curves(N, q, n_treatment, n_control, y_treatment, y_control, z):
    # 输入为完整前缀统计和逐行 HT 权重；返回 Lift、Uplift Gain、Traditional Qini、HT Gain。
    # L(q_k)=Y_T/N_T-Y_C/N_C；缺任何一臂时均值差无定义，不填零。
    has_both = (n_treatment > 0) & (n_control > 0)
    lift = np.full(N + 1, np.nan)
    lift[has_both] = (y_treatment[has_both] / n_treatment[has_both]
                      - y_control[has_both] / n_control[has_both])

    # G(q_k)=(k/N)L(q_k)：Top-k 全部处理的估计增量次数除以 N。
    gain_plugin = q * lift
    gain_plugin[0] = 0.0

    # Q(q_k)=[Y_T-Y_C*N_T/N_C]/N：折算到实际 Treatment 人数。
    # 保留原来的先乘后除顺序；Control-only 前缀为零，缺 Control 时为 NaN。
    has_control = n_control > 0
    qini_traditional = np.full(N + 1, np.nan)
    qini_traditional[has_control] = (
        y_treatment[has_control]
        - y_control[has_control] * n_treatment[has_control] / n_control[has_control]
    ) / N

    # G_HT(q_k)=sum(Z_i, i<=k)/N；累计逐行权重，保持浮点归约顺序。
    gain_ht = np.r_[0.0, np.cumsum(z)] / N
    return lift, gain_plugin, qini_traditional, gain_ht


def evaluate_ranking(y, t, scores, e: float, seed: int = 42, include_curves: bool = True) -> dict:
    #-------------
    # 功能：共享一次排序和前缀计算，生成排序指标与完整 Gain 曲线。
    # 输入参数：
    #     - y：二元结果变量。
    #     - t：二元处理分配变量。
    #     - scores：与样本对齐的策略分数。
    #     - e：用于 HT 的固定处理分配概率。
    #     - seed：同分排序随机种子。
    #     - include_curves：是否生成曲线 DataFrame。
    # 输出参数：
    #     - metrics：样本数、AUUC、Qini 和 HT-ATE。
    #     - curves：包含原点及所有 Top-k 的曲线表，关闭时为 None。
    #-------------
    # Step 1. 校验输入并按分数降序排列；同分使用与结果无关的固定随机优先级。
    y, t = _binary_inputs_check(y, t)
    N = len(y)
    scores = _scores_check(scores, N)
    tie_break = np.random.default_rng(seed).random(N)
    order = np.lexsort((tie_break, -scores))
    y, t = y[order], t[order]

    # Step 2. 计算每个 Top-k 的覆盖率、分组人数、转化数和实际处理比例。
    k = np.arange(N + 1)
    q = k / N
    n_treatment = np.r_[0, np.cumsum(t, dtype=np.int64)]
    n_control = k - n_treatment
    y_treatment = np.r_[0.0, np.cumsum(t * y)]
    y_control = np.r_[0.0, np.cumsum((1 - t) * y)]
    e_k = np.divide(n_treatment, k, out=np.full(N + 1, np.nan), where=k > 0)

    # Step 3. 计算 Lift 和三种 Gain。以下 Y、N 的下标分别表示前缀内的处理组和对照组。
    # lift = Y_T/N_T - Y_C/N_C：Top-k 内两组的转化率之差。
    # gain_plugin = q * lift：按 Top-k 内实际处理比例 e_k 外推至所选全体。
    # qini_traditional = (Y_T - Y_C * N_T/N_C) / N：只折算至前缀内实际处理人数。
    # gain_ht = (Y_T/e - Y_C/(1-e)) / N：用固定 e 加权，实际按排序逐行累计 HT 权重。
    lift, gain_plugin, qini_traditional, gain_ht = _gain_curves(
        N, q, n_treatment, n_control, y_treatment, y_control, _ht_outcome(y, t, e)
    )

    # Step 4. 对完整前缀作梯形积分：auuc_ht = ∫ G_HT(q) dq，auuc_plugin = ∫ G(q) dq。
    # ate_ht = G_HT(1)，qini_ht = auuc_ht - G_HT(1)/2。
    # auuc_plugin_centered = auuc_plugin - G(1)/2，传统 Qini Score = ∫ Q(q) dq - Q(1)/2。
    # 三种积分都从原点开始；Uplift Gain 需要两组，传统 Qini 保留仅有对照组的零值前缀。
    metrics = {
        "n": N,
        **_ranking_areas(N, q, n_treatment, n_control, lift, gain_plugin, qini_traditional, gain_ht),
    }

    # Step 5. 直接整理已有数组；只需要指标时跳过曲线表的构造。
    curves = None
    if include_curves:
        curves = pd.DataFrame({
            "fraction": q,
            "n_selected": k,
            "n_treatment": n_treatment,
            "n_control": n_control,
            "responses_treatment": y_treatment,
            "responses_control": y_control,
            "treatment_fraction": e_k,
            "uplift_rate": lift,
            "gain_plugin": gain_plugin,
            "qini_traditional": qini_traditional,
            "gain_ht": gain_ht,
            "random_gain_ht": q * gain_ht[-1],
        })
        curves.attrs.update({"assignment_probability": float(e), "tie_break_seed": seed})

    # Step 6. 一次返回指标与曲线。
    return {"metrics": metrics, "curves": curves}


def _ranking_areas(N, q, n_treatment, n_control, lift, gain_plugin, qini_traditional, gain_ht):
    # 输入为完整前缀及其曲线数组；输出普通评价和 Bootstrap 共用的六项排序统计。
    # Uplift Gain 需要两组均存在；传统 Qini 保留仅有对照组的零值节点。
    # 跳过未定义前缀后，分别用原点连接首个有效节点。
    has_control = n_control > 0
    has_both = (n_treatment > 0) & has_control
    # HT 的横轴等距，保留先求和再除以 N 的顺序。
    auuc_ht = float(np.sum((gain_ht[:-1] + gain_ht[1:]) * 0.5) / N)
    ate_ht = float(gain_ht[-1])
    qini_ht = auuc_ht - ate_ht / 2
    auuc_plugin = _area(np.r_[0.0, q[has_both]], np.r_[0.0, gain_plugin[has_both]])
    auuc_plugin_centered = auuc_plugin - float(lift[-1]) / 2
    qini_area = _area(np.r_[0.0, q[has_control]], np.r_[0.0, qini_traditional[has_control]])
    qini_score_traditional = qini_area - float(qini_traditional[-1]) / 2
    return {
        "auuc_ht": auuc_ht,
        "qini_ht": qini_ht,
        "ate_ht": ate_ht,
        "auuc_plugin": auuc_plugin,
        "auuc_plugin_centered": auuc_plugin_centered,
        "qini_score_traditional": qini_score_traditional,
    }

def _area(x, values):
    # 按完整节点计算梯形面积；调用方负责原点和未定义区间。
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))


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
    y, t = _binary_inputs_check(y, t)
    z = _ht_outcome(y, t, e)
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
    orders = []
    for name in names:
        scores = _scores_check(score_dict[name], n)
        # 每个策略使用同一组同分优先级，与重抽样随机流分离。
        tie_break = np.random.default_rng(seed).random(n)
        orders.append(np.lexsort((tie_break, -scores)))
    # 臂内有放回抽样始终保留总人数，因此横坐标和逐行 HT 权重只需计算一次。
    k = np.arange(n + 1)
    q = k / n

    def values(order):
        # 只按既定顺序展开抽样记录，绝不重新拟合或重新随机破同分。
        # 复用同一套 Gain 公式和梯形积分，省去 Bootstrap 中不必要的排序、校验和 DataFrame。
        y_ordered, t_ordered = y[order], t[order]
        n_treatment = np.r_[0, np.cumsum(t_ordered, dtype=np.int64)]
        n_control = k - n_treatment
        y_treatment = np.r_[0.0, np.cumsum(t_ordered * y_ordered)]
        y_control = np.r_[0.0, np.cumsum((1 - t_ordered) * y_ordered)]
        lift, gain_plugin, qini_traditional, gain_ht = _gain_curves(
            n, q, n_treatment, n_control, y_treatment, y_control,
            z[order],
        )
        areas = _ranking_areas(
            n, q, n_treatment, n_control, lift, gain_plugin, qini_traditional, gain_ht
        )
        return [areas[metric] for metric in metrics]

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
    # 按重抽样次数展开 Top-k 的累计 HT 贡献，边界记录可只取部分次数。
    cumulative_n = np.cumsum(counts_sorted)
    cumulative_z = np.cumsum(counts_sorted * z_sorted)
    selected = np.zeros(len(k), dtype=float)
    positive = k > 0
    boundary = np.searchsorted(cumulative_n, k[positive], side="left")
    before_n = np.where(boundary > 0, cumulative_n[np.maximum(boundary - 1, 0)], 0)
    before_z = np.where(boundary > 0, cumulative_z[np.maximum(boundary - 1, 0)], 0.0)
    selected[positive] = before_z + (k[positive] - before_n) * z_sorted[boundary]
    return selected
