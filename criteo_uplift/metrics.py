#-------------
# 纯排序指标公式与固定评分的配对 Bootstrap；不训练模型或选择策略。
#-------------
from collections.abc import Mapping

import numpy as np
import pandas as pd
from .input_checks import check_binary_vector

RANKING_METRIC_DESCRIPTION = (
    "按模型预测分数降序排列用户，横轴 q=k/N 表示 Top-k 覆盖比例。"
    "计算每个 Top-k 的累计人数、转化数、Lift 及三种 Gain。"

    "Uplift Gain：使用 Top-k 实际分组比例，估计处理全部 k 人时的总体增量转化率。"
    "Traditional Qini：估计 Top-k 内实际 Treatment 的 N_T(k) 人对应的总体增量转化率。"
    "HT Gain：使用固定分配概率 e 进行逆概率加权，估计处理全部 k 人时的总体增量转化率。"

    "AUUC 对 Uplift Gain 积分，Qini Score 对传统 Qini 积分并减去其终点值的一半。"

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


GAIN_METRICS = ("gain_plugin", "qini_traditional", "gain_ht")
RANKING_METRICS = ("auuc", "qini_score", *GAIN_METRICS)


def _gain_curves(N, q, n_treatment, n_control, y_treatment, y_control, z, *, metric=None):
    # 复用三种 Gain 的公式；Bootstrap 只构造主指标所需的数值。
    if metric in (None, "gain_plugin"):
        has_both = (n_treatment > 0) & (n_control > 0)
        lift = np.full(q.shape, np.nan)
        lift[has_both] = (y_treatment[has_both] / n_treatment[has_both]
                          - y_control[has_both] / n_control[has_both])
        gain_plugin = q * lift
        gain_plugin[q == 0] = 0.0
        if metric == "gain_plugin":
            return gain_plugin
    if metric in (None, "qini_traditional"):
        has_control = n_control > 0
        qini = np.full(q.shape, np.nan)
        qini[has_control] = (y_treatment[has_control]
                             - y_control[has_control] * n_treatment[has_control]
                             / n_control[has_control]) / N
        if metric == "qini_traditional":
            return qini
    gain_ht = z / N
    return gain_ht if metric == "gain_ht" else (lift, gain_plugin, qini, gain_ht)


def _area(q, gain, *, centered=False):
    # 在原点与有定义的完整前缀上积分。仅有对照组的 Qini 零值点参与积分。
    valid = (q > 0) & np.isfinite(gain)
    x = np.r_[0., q[valid]]
    values = np.r_[0., gain[valid]]
    area = float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))
    return area - float(gain[-1]) / 2 if centered else area


def evaluate_ranking(
    y, t, score_dict, e, *, budgets=(.10, .20, .30, 1.00),
    primary_metric="gain_ht", primary_budget=.20, n_bootstrap=200,
    seed=42, include_curves=True, include_random=False,
) -> dict:
    #-------------
    # 功能：一次评价全部评分策略，生成五项指标，并可对唯一主指标做配对 Bootstrap。
    # 输入参数：
    #     - y：二元结果。
    #     - t：二元处理分配。
    #     - score_dict：策略名称及对应评分，同分按共享的随机优先级排序。
    #     - e：固定的处理分配概率。
    #     - budgets：需要报告的覆盖率。
    #     - primary_metric：五项正式指标中的唯一主指标。
    #     - primary_budget：Gain 主指标的选型覆盖率。
    #     - n_bootstrap：重抽样次数，0 表示关闭。
    #     - seed：排序和重抽样的随机种子。
    #     - include_curves：是否返回绘图节点，每策略最多 501 行。
    # 输出参数：
    #     - metrics：按策略和预算组织的五项点估计，原始标度及 HT 每十万人值。
    #     - curves：绘图表或 None，积分始终使用完整前缀。
    #     - bootstrap：主指标的汇总、配对区间、获胜频率，关闭时为 None。
    #-------------
    # 1. 校验输入和主指标，Gain 主预算自动进入报告预算。
    y, t = _binary_inputs_check(y, t)
    z = _ht_outcome(y, t, e)
    if not isinstance(score_dict, Mapping) or not score_dict or any(
        not isinstance(name, str) or not name or name == "random" for name in score_dict
    ):
        raise ValueError("score_dict 必须是非空评分映射，random 为保留名称")
    n = len(y)
    names = list(score_dict)
    scores = [_scores_check(score_dict[name], n) for name in names]
    if primary_metric not in RANKING_METRICS:
        raise ValueError(f"primary_metric 必须属于 {RANKING_METRICS}")
    if not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, bool) or n_bootstrap == 1 or n_bootstrap < 0:
        raise ValueError("n_bootstrap 必须为 0 或至少 2 的整数")
    budget = np.asarray(budgets, dtype=float)
    if budget.ndim != 1 or not len(budget) or not np.isfinite(budget).all() or np.any((budget < 0) | (budget > 1)):
        raise ValueError("budgets 必须为非空一维覆盖率序列，取值介于 0 和 1")
    gain_primary = primary_metric in GAIN_METRICS
    if gain_primary:
        if not np.isscalar(primary_budget) or not np.isfinite(primary_budget) or not 0 < primary_budget <= 1:
            raise ValueError("Gain 主指标的 primary_budget 必须属于 (0, 1]")
        budget = np.r_[budget, primary_budget]
    budget = np.unique(budget)
    selected = np.floor(budget * n + 1e-12).astype(int)
    k = np.arange(n + 1)
    q = k / n
    plot_nodes = np.unique(np.linspace(0, n, min(501, n + 1)).astype(int))
    tie_break = np.random.default_rng(seed).random(n)
    orders, metric_rows, curve_tables = [], [], []
    ht_endpoint = float(z.mean())

    for name, score in zip(names, scores):
        # 2. 固定排序与累计统计。每个评分只排序一次，Bootstrap 复用其顺序。
        order = np.lexsort((tie_break, -score))
        orders.append(order)
        yy, tt = y[order], t[order]
        nt = np.r_[0, np.cumsum(tt, dtype=np.int64)]
        nc = k - nt
        yt = np.r_[0., np.cumsum(tt * yy)]
        yc = np.r_[0., np.cumsum((1 - tt) * yy)]
        lift, plugin, qini, ht = _gain_curves(n, q, nt, nc, yt, yc, np.r_[0., np.cumsum(z[order])])
        ht[-1] = ht_endpoint

        # 3. 两项面积使用全部节点。Qini Score 扣除终点参考线下面积。
        auuc = _area(q, plugin)
        qini_score = _area(q, qini, centered=True)

        # 4. 预算从现有 Gain 数组按 floor(qN) 取值，HT 另保留每十万人标度。
        for fraction, index in zip(budget, selected):
            metric_rows.append({
                "policy": name, "budget": float(fraction), "n_selected": int(index),
                "auuc": auuc, "qini_score": qini_score, "gain_plugin": plugin[index],
                "qini_traditional": qini[index], "gain_ht": ht[index],
                "gain_ht_per_100k": ht[index] * 100_000,
            })
        if include_curves:
            ix = plot_nodes
            curve_tables.append(pd.DataFrame({
                "policy": name, "fraction": q[ix], "n_selected": k[ix],
                "n_treatment": nt[ix], "n_control": nc[ix],
                "responses_treatment": yt[ix], "responses_control": yc[ix],
                "uplift_rate": lift[ix], "gain_plugin": plugin[ix],
                "qini_traditional": qini[ix], "gain_ht": ht[ix],
                "random_gain_ht": q[ix] * ht_endpoint,
            }))

    del yy, tt, nt, nc, yt, yc, lift, plugin, qini, ht
    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_tables, ignore_index=True) if include_curves else None
    bootstrap = None
    if n_bootstrap:
        # 5. 每轮共用同一份分层抽样。预算 Gain 只计算指定节点，面积只计算对应曲线。
        boot_budgets = budget if gain_primary else [np.nan]
        boot_k = selected if gain_primary else k
        boot_q = boot_k / n
        curve_metric = {"auuc": "gain_plugin", "qini_score": "qini_traditional"}.get(primary_metric, primary_metric)
        draws = np.empty((n_bootstrap, len(names) + 1, len(boot_budgets)))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
        arm1, arm0 = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
        for b in range(n_bootstrap):
            draw1 = arm1[rng.integers(len(arm1), size=len(arm1))]
            draw0 = arm0[rng.integers(len(arm0), size=len(arm0))]
            counts = np.bincount(draw1, minlength=n) + np.bincount(draw0, minlength=n)
            if primary_metric == "gain_ht":
                endpoint = np.dot(counts, z) / n
            else:
                endpoint = y[draw1].mean() - y[draw0].mean()
                if curve_metric == "qini_traditional":
                    endpoint *= len(arm1) / n
            if gain_primary:
                draws[b, -1] = boot_q * endpoint
            else:
                draws[b, -1] = endpoint / 2 if primary_metric == "auuc" else 0.

            for j, order in enumerate(orders):
                expanded = np.repeat(order, counts[order])[:boot_k[-1]]
                if primary_metric == "gain_ht":
                    gain = _gain_curves(n, boot_q, None, None, None, None,
                                        np.r_[0., np.cumsum(z[expanded])][boot_k], metric=curve_metric)
                    gain[boot_k == n] = endpoint
                else:
                    tt, yy = t[expanded], y[expanded]
                    nt = np.r_[0, np.cumsum(tt, dtype=np.int64)][boot_k]
                    yt = np.r_[0., np.cumsum(tt * yy)][boot_k]
                    yc = np.r_[0., np.cumsum((1 - tt) * yy)][boot_k]
                    gain = _gain_curves(n, boot_q, nt, boot_k - nt, yt, yc, None, metric=curve_metric)
                draws[b, j] = gain if gain_primary else _area(q, gain, centered=primary_metric == "qini_score")

        # 6. 单策略区间使用该策略有效轮次，配对区间只使用共同有效轮次。
        # 获胜频率只统计全部候选均有定义的轮次，并按输入顺序处理同分。
        points = (metrics[primary_metric].to_numpy().reshape(len(names), len(budget))
                  if gain_primary else metrics.groupby("policy", sort=False)[primary_metric].first().to_numpy()[:, None])
        all_names = [*names, "random"]
        if primary_metric == "gain_ht":
            random_point = boot_q * ht_endpoint
        elif primary_metric == "qini_score":
            random_point = np.array([0.])
        else:
            endpoint = (y[arm1].mean() - y[arm0].mean())
            if primary_metric == "qini_traditional":
                endpoint *= len(arm1) / n
            random_point = boot_q * endpoint if gain_primary else np.array([endpoint / 2])
        points = np.vstack((points, random_point))
        summary, pairwise, frequencies = [], [], []
        for i, fraction in enumerate(boot_budgets):
            candidates = draws[:, :-1, i]
            complete = np.isfinite(candidates).all(axis=1)
            winners = np.argmax(candidates[complete], axis=1)
            for j, name in enumerate(all_names):
                values = draws[:, j, i]
                valid = values[np.isfinite(values)]
                lo, hi = np.quantile(valid, [.025, .975]) if valid.size else (np.nan, np.nan)
                row = {"policy": name, "metric": primary_metric, "budget": fraction}
                count = int(np.count_nonzero(winners == j))
                frequency = count / len(winners) if len(winners) and name != "random" else np.nan
                summary.append({**row, "point_estimate": points[j, i],
                                "bootstrap_mean": valid.mean() if valid.size else np.nan,
                                "ci_low": lo, "ci_high": hi, "n_bootstrap": n_bootstrap,
                                "valid_replicates": len(valid), "win_frequency": frequency})
                if name != "random":
                    frequencies.append({**row, "wins": count, "valid_replicates": len(winners),
                                        "win_frequency": frequency})
                for r, reference in enumerate(all_names):
                    if r == j:
                        continue
                    difference = values - draws[:, r, i]
                    paired = difference[np.isfinite(difference)]
                    lo, hi = np.quantile(paired, [.025, .975]) if paired.size else (np.nan, np.nan)
                    pairwise.append({**row, "reference": reference,
                                     "difference": points[j, i] - points[r, i],
                                     "bootstrap_mean_difference": paired.mean() if paired.size else np.nan,
                                     "ci_low": lo, "ci_high": hi, "valid_replicates": len(paired)})
        bootstrap = {"summary": pd.DataFrame(summary), "pairwise": pd.DataFrame(pairwise),
                     "selection_frequency": pd.DataFrame(frequencies)}
    if include_random:
        # Analytic random reference, not a fitted/random-score candidate.
        full_plugin = float(y[t == 1].mean() - y[t == 0].mean())
        random_rows = pd.DataFrame({
            'policy': 'random', 'budget': budget, 'n_selected': selected,
            'gain_ht': selected / n * ht_endpoint,
            'gain_ht_per_100k': selected / n * ht_endpoint * 100_000,
            'gain_plugin': selected / n * full_plugin,
            'qini_traditional': selected / n * full_plugin * t.mean(),
            'auuc': full_plugin / 2, 'qini_score': 0.,
        })
        metrics = pd.concat([metrics, random_rows], ignore_index=True)
    # Place the sole business selection metric first in exported comparison tables.
    metrics = metrics[['policy', 'budget', 'n_selected', 'gain_ht', 'gain_ht_per_100k',
                       'gain_plugin', 'qini_traditional', 'auuc', 'qini_score']]
    return {"metrics": metrics, "curves": curves, "bootstrap": bootstrap}
