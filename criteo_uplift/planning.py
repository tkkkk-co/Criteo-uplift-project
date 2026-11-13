# 在明确的假设下估算策略 A/B 实验样本量和成本收益，不代表真实线上收益。

from math import ceil, floor, isfinite, sqrt
from statistics import NormalDist

_NORMAL = NormalDist()


def plan_policy_ab(
    p0, absolute_mde, allocation_ratio=1.0, budget=0.2, alpha=0.05, power=0.8
):
    #-------------
    # 函数名：plan_policy_ab
    # 功能：根据基线转化率、绝对 MDE、分配比例、显著性水平和功效估算策略 A/B 实验样本量。
    # 输入参数：
    #     - p0：假设的基线策略总体转化率。
    #     - absolute_mde：需要检测的绝对转化率差异。
    #     - allocation_ratio：策略 A 与策略 B 的样本分配比例。
    #     - budget：假设的触达比例。
    #     - alpha：双侧检验显著性水平。
    #     - power：目标统计功效。
    # 输出参数：
    #     - 规划结果字典：包含两组样本量、总样本量、触达人数及近似实际功效。
    #-------------
    values = (p0, absolute_mde, allocation_ratio, budget, alpha, power)
    if not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
        raise ValueError('输入必须为有限数值标量')
    p1, r = p0 + absolute_mde, allocation_ratio
    if not (
        0 < p0 < p1 < 1
        and r > 0
        and 0 < budget <= 1
        and 0 < alpha < 0.5
        and 0.5 < power < 1
    ):
        raise ValueError(
            '参数需满足 0<p0<p0+MDE<1、ratio>0、budget 属于 (0,1]、alpha 属于 (0,.5)、power 属于 (.5,1)'
        )
    pbar = (p0 + r * p1) / (1 + r)
    za, zb = _NORMAL.inv_cdf(1 - alpha / 2), _NORMAL.inv_cdf(power)
    null_v = pbar * (1 - pbar) * (1 + 1 / r)
    alt_v = p0 * (1 - p0) + p1 * (1 - p1) / r
    n0 = ceil((za * sqrt(null_v) + zb * sqrt(alt_v)) ** 2 / absolute_mde**2)
    n1 = ceil(r * n0)
    pooled = (n0 * p0 + n1 * p1) / (n0 + n1)
    null_se = sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    alt_se = sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    achieved = (
        1
        - _NORMAL.cdf((za * null_se - absolute_mde) / alt_se)
        + _NORMAL.cdf((-za * null_se - absolute_mde) / alt_se)
    )
    return {
        'p0_assumed': p0,
        'p1_assumed': p1,
        'absolute_mde': absolute_mde,
        'relative_mde': absolute_mde / p0,
        'alpha_two_sided': alpha,
        'target_power': power,
        'allocation_ratio_A_over_B': r,
        'n_policy_B': n0,
        'n_policy_A': n1,
        'n_total': n0 + n1,
        'contact_fraction_each_policy': budget,
        'contacts_B': floor(n0 * budget),
        'contacts_A': floor(n1 * budget),
        'approximate_achieved_power': achieved,
        'assumption': 'Hypothetical policy CVRs on all eligible users; not inferred online rates from Criteo.',
    }


def cost_value_scenario(
    incremental_per_100k,
    *,
    budget,
    value_per_conversion,
    contact_cost,
    eligible_n=100_000,
    ci_low=None,
    ci_high=None,
):
    #-------------
    # 函数名：cost_value_scenario
    # 功能：将每十万名符合条件人群的增量结果换算为假设收益、触达成本、净价值和条件性区间。
    # 输入参数：
    #     - incremental_per_100k：每十万名符合条件人群的增量结果数。
    #     - budget：假设的触达比例。
    #     - value_per_conversion：每次增量转化的假设贡献价值。
    #     - contact_cost：每次触达的假设单位成本。
    #     - eligible_n：符合触达条件的总人数。
    #     - ci_low：可选的增量效果置信区间下界。
    #     - ci_high：可选的增量效果置信区间上界。
    # 输出参数：
    #     - 经济情景字典：包含增量转化、收益、成本、净值、条件性区间及盈亏平衡触达成本。
    #-------------
    values = (incremental_per_100k, budget, value_per_conversion, contact_cost)
    if not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
        raise ValueError('输入必须为有限数值标量')
    if not (0 < budget <= 1 and value_per_conversion >= 0 and contact_cost >= 0):
        raise ValueError('budget 必须属于 (0,1]，value 和 cost 必须大于等于 0')
    if (
        not isinstance(eligible_n, int)
        or isinstance(eligible_n, bool)
        or eligible_n <= 0
    ):
        raise ValueError('eligible_n 必须为正整数')
    if (ci_low is None) != (ci_high is None):
        raise ValueError('效应置信区间的两个端点必须同时提供或同时省略')
    if ci_low is not None and (
        not isfinite(ci_low) or not isfinite(ci_high) or ci_low > ci_high
    ):
        raise ValueError('置信区间端点必须为有限值，且下界不能大于上界')
    contacts = floor(eligible_n * budget + 1e-12)
    scale = eligible_n / 100_000
    benefit = incremental_per_100k * scale * value_per_conversion
    cost = contacts * contact_cost
    return {
        'incremental_per_100k': incremental_per_100k,
        'eligible_n': eligible_n,
        'budget': budget,
        'contacts': contacts,
        'value_per_conversion_assumed': value_per_conversion,
        'contact_cost_assumed': contact_cost,
        'incremental_conversions': incremental_per_100k * scale,
        'gross_incremental_value': benefit,
        'contact_cost_total': cost,
        'net_incremental_value': benefit - cost,
        'net_ci_low_conditional': (
            None if ci_low is None else ci_low * scale * value_per_conversion - cost
        ),
        'net_ci_high_conditional': (
            None if ci_high is None else ci_high * scale * value_per_conversion - cost
        ),
        'break_even_contact_cost': benefit / contacts if contacts else None,
        'assumption': 'Hypothetical constant value/cost, fixed effect transport; no actual revenue or ROI claim.',
    }


DEFAULT_PLANNING_ASSUMPTIONS = {
    "p0": 0.003, "absolute_mdes": [0.00015, 0.0003, 0.0006],
    "allocation_ratio": 1.0, "alpha": 0.05, "power": 0.8,
    "values_per_conversion": [1.0, 5.0, 10.0],
    "contact_costs": [0.001, 0.01, 0.05], "eligible_n": 100_000,
}


def run_planning(evaluation, selection, out, *, assumptions=None):
    #-------------
    # 函数名：run_planning
    # 功能：读取冻结策略在主预算下的 Test 结果，按明确的业务假设生成 A/B 样本量规划和经济情景文件。
    # 输入参数：
    #     - evaluation：Test 的统一评价结果，业务情景使用 HT Gain。
    #     - selection：Validation 阶段冻结的策略名称和主预算。
    #     - out：本次实验的 business 输出目录。
    #     - assumptions：可选的业务假设字典；未提供时使用 planning.py 中的教学情景。
    # 输出参数：
    #     - 无；保存 assumptions.json、hypothetical_ab_plans.csv 和 hypothetical_cost_scenarios.csv。
    #-------------
    import pandas as pd
    import numpy as np
    from .pipeline_io import safe_output, write_result

    settings = dict(DEFAULT_PLANNING_ASSUMPTIONS if assumptions is None else assumptions)
    if set(settings) != set(DEFAULT_PLANNING_ASSUMPTIONS):
        raise ValueError(f"规划假设必须恰好包含这些字段：{list(DEFAULT_PLANNING_ASSUMPTIONS)}")
    for key in ('absolute_mdes', 'values_per_conversion', 'contact_costs'):
        if not isinstance(settings[key], (list, tuple)) or not settings[key]:
            raise ValueError(f'{key} 必须为非空列表')
    budget = selection['primary_budget']
    policy = selection['selected_policy']
    metrics = evaluation["metrics"]
    row = metrics[(metrics.policy == policy) & metrics.budget.eq(budget)].iloc[0]
    ci_low = ci_high = None
    if selection["primary_metric"] == "gain_ht":
        summary = evaluation["bootstrap"]["summary"]
        interval = summary[(summary.policy == policy) & summary.budget.eq(budget)].iloc[0]
        if np.isfinite([interval.ci_low, interval.ci_high]).all():
            ci_low, ci_high = interval.ci_low * 100_000, interval.ci_high * 100_000
    plans = pd.DataFrame([
        plan_policy_ab(settings['p0'], delta, settings['allocation_ratio'], budget,
                       settings['alpha'], settings['power'])
        for delta in settings['absolute_mdes']
    ])
    scenarios = pd.DataFrame([
        cost_value_scenario(float(row.gain_ht_per_100k), budget=budget,
                            value_per_conversion=value, contact_cost=cost,
                            eligible_n=settings['eligible_n'], ci_low=ci_low, ci_high=ci_high)
        for value in settings['values_per_conversion'] for cost in settings['contact_costs']
    ])
    out = safe_output(out)
    write_result({"hypothetical": True, "parameters": settings,
                  "frozen_selected_policy": selection['selected_policy'], "budget": budget,
                  "effect_source": "../evaluation/test_metrics.csv",
                  "conditional_ht_interval": ci_low is not None,
                  "interpretation": "假设固定价值、触达成本与效果可迁移；不是 Criteo 真实线上收入或 ROI。"},
                 out / 'assumptions.json')
    write_result(plans, out / 'hypothetical_ab_plans.csv')
    write_result(scenarios, out / 'hypothetical_cost_scenarios.csv')
