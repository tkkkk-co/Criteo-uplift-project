# 在明确的假设下估算策略 A/B 实验样本量和成本收益，不代表真实线上收益。
# Notebook 06 使用本模块；所有价值/成本输入均为显式假设，不能当作真实财务数据。

from math import ceil, floor, isfinite, sqrt
from statistics import NormalDist

_NORMAL = NormalDist()


#-----------------------------------------------------------------------------------------------------------------------
# 函数：plan_policy_ab —— 估算假设策略对比所需的样本量。
# 输入：
#     p0：假设的基线策略总体转化率。
#     absolute_mde：希望检测到的绝对转化率差异。
#     allocation_ratio：策略 A 与策略 B 的分配人数比例。
#     budget：本次允许触达的人群比例。
#     alpha：双侧检验的显著性水平。
#     power：目标统计功效。
# 输出：
#     规划结果：两组样本量、总人数、触达人数和近似功效等假设性结果。
# 说明：基于独立二分类结果的近似规划；转化率指全部符合条件人群，预算不会额外改变所需效应差。
#-----------------------------------------------------------------------------------------------------------------------
def plan_policy_ab(
    p0, absolute_mde, allocation_ratio=1.0, budget=0.2, alpha=0.05, power=0.8
):
    values = (p0, absolute_mde, allocation_ratio, budget, alpha, power)
    if not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
        raise ValueError('Inputs must be finite numeric scalars')
    p1, r = p0 + absolute_mde, allocation_ratio
    if not (
        0 < p0 < p1 < 1
        and r > 0
        and 0 < budget <= 1
        and 0 < alpha < 0.5
        and 0.5 < power < 1
    ):
        raise ValueError(
            'Require 0<p0<p0+MDE<1, ratio>0, budget in (0,1], alpha in (0,.5), power in (.5,1)'
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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：cost_value_scenario —— 将增量效果转换为假设成本收益。
# 输入：
#     incremental_per_100k：每十万名符合条件人群的增量结果数。
#     budget：本次允许触达的人群比例。
#     value_per_conversion：假设每次增量转化带来的贡献价值。
#     contact_cost：假设每次触达的单位成本。
#     eligible_n：符合触达条件的总体人数。
#     ci_low：可选的增量效果区间下界。
#     ci_high：可选的增量效果区间上界。
# 输出：
#     情景结果：增量转化、收益、触达成本、净值及条件区间。
# 说明：价值、成本和效果可迁移性均由输入假设给出，不代表已实现的线上收入。
#-----------------------------------------------------------------------------------------------------------------------
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
    values = (incremental_per_100k, budget, value_per_conversion, contact_cost)
    if not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
        raise ValueError('Inputs must be finite numeric scalars')
    if not (0 < budget <= 1 and value_per_conversion >= 0 and contact_cost >= 0):
        raise ValueError('Require budget in (0,1], value and cost >=0')
    if (
        not isinstance(eligible_n, int)
        or isinstance(eligible_n, bool)
        or eligible_n <= 0
    ):
        raise ValueError('eligible_n must be a positive integer')
    if (ci_low is None) != (ci_high is None):
        raise ValueError('Supply both effect CI endpoints or neither')
    if ci_low is not None and (
        not isfinite(ci_low) or not isfinite(ci_high) or ci_low > ci_high
    ):
        raise ValueError('CI endpoints must be finite and ordered')
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
