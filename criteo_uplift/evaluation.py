#-------------
# ATE、策略价值、分组校准与冻结模型的 Test 评价。
#-------------
from math import sqrt
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from .config import FEATURES
from .pipeline_io import safe_output, write_result
from .models import LEARNERS
from .metrics import (_binary_inputs, _scores, _ht_outcome, _rank, _prefix_values,
                      ranking_metrics, ranking_confidence_intervals, uplift_curve)

DEFAULT_BUDGETS = (0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
_Z_975 = 1.959963984540054

def _normal_pvalue(z):
    #-------------
    # 函数名：_normal_pvalue
    # 功能：根据标准正态统计量计算双侧尾部概率，并保留极小概率的对数值和文本表示。
    # 输入参数：
    #     - z：标准正态检验统计量。
    # 输出参数：
    #     - 三元组：浮点 p 值、log(p) 和便于展示的 p 值文本。
    #-------------
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

def _ate_summary_from_counts(n0, n1, r0, r1):
    #-------------
    # 函数名：_ate_summary_from_counts
    # 功能：根据两组人数和正例人数计算差均值 ATE、标准误、置信区间及 pooled 零假设检验。
    # 输入参数：
    #     - n0：Control 组人数。
    #     - n1：Treatment 组人数。
    #     - r0：Control 组结果变量正例人数。
    #     - r1：Treatment 组结果变量正例人数。
    # 输出参数：
    #     - ATE 统计字典：包含两组响应率、ATE、标准误、95% CI、p 值及相对增幅。
    #-------------
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (n0, n1, r0, r1)):
        raise ValueError("审计计数必须为有效整数")
    if not (n0 >= 0 and n1 >= 0 and n0 + n1 > 0 and 0 <= r0 <= n0 and 0 <= r1 <= n1):
        raise ValueError("审计计数必须为有效整数，且不能超过对应处理臂人数")
    # 校准小分箱可能缺一组，此时保留 NaN，不伪造可估计的效果。
    p0, p1 = r0 / n0 if n0 else np.nan, r1 / n1 if n1 else np.nan
    ate = p1 - p0
    se = sqrt(p1 * (1 - p1) / (n1 - 1) + p0 * (1 - p0) / (n0 - 1)) if min(n0, n1) > 1 else np.nan
    pooled = (r0 + r1) / (n0 + n1)
    null_se = sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1)) if min(n0, n1) else np.nan
    z = ate / null_se if null_se > 0 else (0.0 if null_se == 0 and ate == 0 else np.nan)
    p_value, log_p_value, p_value_text = _normal_pvalue(z)
    return {
        "n": n0 + n1, "n_treatment": n1, "n_control": n0,
        "response_treatment": p1, "response_control": p0, "ate": ate,
        "standard_error": se, "ci_low": ate - _Z_975 * se, "ci_high": ate + _Z_975 * se,
        "pooled_null_standard_error": null_se, "p_value": p_value,
        "log_p_value": log_p_value, "p_value_text": p_value_text,
        "relative_uplift": ate / p0 if p0 > 0 else np.nan,
    }


def ate_summary(y, t):
    #-------------
    # 函数名：ate_summary
    # 功能：从逐行二元结果变量和处理分配变量计算 ATE 统计摘要。
    # 输入参数：
    #     - y：逐行观测到的 0/1 结果变量。
    #     - t：逐行 0/1 Treatment 分配变量。
    # 输出参数：
    #     - ATE 统计字典：包含两组响应率、ATE、标准误、95% CI、p 值及相对增幅。
    #-------------
    y, t = _binary_inputs(y, t)
    return _ate_summary_from_counts(int((t == 0).sum()), int((t == 1).sum()),
                                    int(y[t == 0].sum()), int(y[t == 1].sum()))


def ate_summary_from_audit(audit, outcome):
    #-------------
    # 函数名：ate_summary_from_audit
    # 功能：直接使用全量审计中的分组人数和正例计数计算全量 ATE，不重新读取原始数据。
    # 输入参数：
    #     - audit：Preparation 生成的全量审计字典。
    #     - outcome：结果指标名称，支持 conversion 或 visit。
    # 输出参数：
    #     - 与 ate_summary 相同字段结构的 ATE 统计字典。
    #-------------
    if outcome not in ("conversion", "visit"):
        raise ValueError("outcome 必须为 conversion 或 visit")
    groups = audit["groups"]
    n0, n1 = (groups[str(arm)]["n"] for arm in (0, 1))
    r0, r1 = (groups[str(arm)][outcome]["positive"] for arm in (0, 1))
    if n0 <= 0 or n1 <= 0:
        raise ValueError("审计计数必须同时包含处理组和对照组")
    return _ate_summary_from_counts(n0, n1, r0, r1)


def evaluate_policies(
    y,
    t,
    score_dict: Mapping[str, Sequence[float]],
    e: float,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    #-------------
    # 函数名：evaluate_policies
    # 功能：在给定预算下比较多种策略的 HT 增量价值，并计算组内配对 Bootstrap 置信区间。
    # 输入参数：
    #     - y：测试/评价样本的二元结果变量。
    #     - t：对应样本的二元处理分配变量。
    #     - score_dict：策略名称到逐行评分向量的映射。
    #     - e：固定处理分配概率，用于 HT 结果加权。
    #     - budgets：需要评价的触达比例序列。
    #     - bootstrap：组内配对 Bootstrap 重抽样次数；0 表示不计算区间。
    #     - seed：排序破同分和 Bootstrap 的随机种子。
    # 输出参数：
    #     - Policy Evaluation DataFrame：每种策略、每个预算的增量值、区间及相对随机/倾向基线的差值。
    #-------------
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
            "budgets 必须为非空序列，各比例需介于 0 和 1 之间"
        )
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap 必须为 0 或大于等于 2 的整数")
    if "random" in score_dict:
        raise ValueError("'random' 为解析随机策略基线的保留名称")
    if any(not isinstance(name, str) or not name for name in score_dict):
        raise ValueError("策略名称必须为非空字符串")
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

def calibration_table(y, t, cate, n_bins: int = 10) -> pd.DataFrame:
    #-------------
    # 函数名：calibration_table
    # 功能：按预测 CATE 从低到高分箱，比较每箱平均预测效应与实际处理组/对照组结果率差。
    # 输入参数：
    #     - y：测试样本的二元结果变量。
    #     - t：测试样本的二元处理分配变量。
    #     - cate：每个样本的预测 CATE 分数。
    #     - n_bins：等人数校准分箱数量。
    # 输出参数：
    #     - 校准结果 DataFrame：每箱的样本量、预测 CATE、观察 ATE 和置信区间。
    #-------------
    y, t = _binary_inputs(y, t)
    cate = _scores(cate, len(y))
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins 必须为正整数")
    order = _rank(-cate, seed=42)
    records = []
    for number, selected in enumerate(
        np.array_split(order, min(n_bins, len(y))), start=1
    ):
        labels, treatment = y[selected], t[selected]
        summary = _ate_summary_from_counts(
            int((treatment == 0).sum()), int((treatment == 1).sum()),
            int(labels[treatment == 0].sum()), int(labels[treatment == 1].sum()),
        )
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

def evaluate_test_set(
    test,
    models,
    *,
    outcome,
    assignment_probability,
    seed,
    bootstrap,
    primary_budget,
    out,
    progress=print,
    full_ate=None,
):
    #-------------
    # 函数名：evaluate_test_set
    # 功能：使用已经冻结的模型在 Test 集上生成 CATE 评分、排序指标、策略价值、校准结果和 ATE，并保存评价产物。
    # 输入参数：
    #     - test：独立 Test DataFrame，包含特征、treatment、结果和 row_id。
    #     - models：已在训练集拟合并冻结的模型字典。
    #     - outcome：本次模型训练和主要排序评价使用的结果列名。
    #     - assignment_probability：用于 HT 评价的固定处理分配概率。
    #     - seed：排序破同分和 Bootstrap 随机种子。
    #     - bootstrap：评价区间的 Bootstrap 重抽样次数。
    #     - primary_budget：Validation 阶段冻结的主要触达预算。
    #     - out：本次实验的 evaluation 输出目录。
    #     - progress：接收评价进度消息的回调函数。
    #     - full_ate：由全量 Audit 计算得到的 ATE 记录列表。
    # 输出参数：
    #     - Test Policy Evaluation DataFrame；同时保存 ATE、曲线、排名、校准、区间和逐行预测缓存。
    #-------------
    out = safe_output(out)
    test_scores = {}
    for name, model in models.items():
        result = LEARNERS[model['learner']](test[FEATURES], fitted=model)
        test_scores[name] = result['cate']
        if name == 't_xgboost':
            propensity = result['mu1']
    test_scores['propensity'] = propensity

    rankings = []
    curves = []
    for name, score in test_scores.items():
        rankings.append(
            {
                "policy": name,
                **ranking_metrics(
                    test[outcome],
                    test.treatment,
                    score,
                    assignment_probability,
                    seed,
                ),
                "mean_score": float(np.mean(score)),
                "negative_fraction": float(np.mean(score < 0)),
            }
        )
        curve = uplift_curve(
            test[outcome],
            test.treatment,
            score,
            assignment_probability,
            seed,
        )
        # 指标使用完整曲线；仅压缩用于绘图的保存表。
        curve = curve.iloc[
            np.unique(np.linspace(0, len(curve) - 1, 501).astype(int))
        ].copy()
        curve.insert(0, "policy", name)
        curves.append(curve)
        if name != "propensity":
            calibration = calibration_table(test[outcome], test.treatment, score)
            write_result(calibration, out / f'calibration_{name}.csv')

    write_result(pd.DataFrame(rankings), out / 'test_ranking.csv')
    write_result(pd.concat(curves), out / 'curves.csv')

    budgets = sorted(set([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, primary_budget]))
    policies = evaluate_policies(
        test[outcome],
        test.treatment,
        test_scores,
        assignment_probability,
        budgets=budgets,
        bootstrap=bootstrap,
        seed=seed,
    )
    write_result(policies, out / 'test_policies.csv')
    progress("计算冻结模型的配对 AUUC / Qini 区间")
    ranking_intervals = ranking_confidence_intervals(
        test[outcome],
        test.treatment,
        test_scores,
        assignment_probability,
        bootstrap=bootstrap,
        seed=seed,
    )
    write_result(ranking_intervals, out / 'ranking_intervals.csv')

    test_ate = [{"scope": "test", "outcome": label, **ate_summary(test[label], test.treatment)}
                for label in ("conversion", "visit") if label in test]
    all_ate = pd.concat([pd.DataFrame(full_ate), pd.DataFrame(test_ate)], ignore_index=True)
    write_result(all_ate, out / 'ate_summary.csv')
    np.savez_compressed(
        (out / 'predictions.npz'),
        row_id=test.row_id.to_numpy(),
        y=test[outcome].to_numpy(),
        t=test.treatment.to_numpy(),
        **test_scores,
    )
    return policies
