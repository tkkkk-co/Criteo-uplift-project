#-------------
# ATE、策略价值与冻结模型的 Test 评价。
#-------------
import json
from math import sqrt
import joblib
import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from .config import FEATURES, POLICY_NAMES
from .pipeline_io import safe_output, write_result
from .models import s_learner, t_learner, x_learner
from .metrics import _binary_inputs_check, evaluate_ranking

DEFAULT_BUDGETS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
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
    # 按处理臂计算响应率和未合并标准误。
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
    y, t = _binary_inputs_check(y, t)
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


def evaluate_test_set(
    test,
    *,
    outcome,
    assignment_probability,
    seed,
    out,
    progress=print,
    full_ate,
):
    #-------------
    # 函数名：evaluate_test_set
    # 功能：使用已经冻结的模型在 Test 集上生成 CATE 评分、排序指标、策略价值和 ATE，并保存评价产物。
    # 输入参数：
    #     - test：独立 Test DataFrame，包含特征、treatment、结果和 row_id。
    #     - outcome：本次模型训练和主要排序评价使用的结果列名。
    #     - assignment_probability：用于 HT 评价的固定处理分配概率。
    #     - seed：排序破同分和 Bootstrap 随机种子。
    #     - out：本次实验的 evaluation 输出目录。
    #     - progress：接收评价进度消息的回调函数。
    #     - full_ate：由全量 Audit 计算得到的 ATE 记录列表。
    # 输出参数：
    #     - 统一评价结果字典。另保存ATE 和逐行预测。
    #-------------
    out = safe_output(out)
    models_out = out.parent / 'models'
    selection = json.loads((models_out / 'selection.json').read_text(encoding='utf-8'))
    if (not selection["frozen_before_test"] or tuple(selection["candidates"]) != POLICY_NAMES
            or selection["selected_policy"] not in POLICY_NAMES):
        raise ValueError("Test 需要有效的 Validation 冻结策略")
    test_scores = {}
    for name in POLICY_NAMES[:-1]:
        model_bundle = joblib.load(models_out / f'{name}.joblib')
        if model_bundle['learner'] == 's':
            result = s_learner(test[FEATURES], model_bundle=model_bundle)
        elif model_bundle['learner'] == 't':
            result = t_learner(test[FEATURES], model_bundle=model_bundle)
        elif model_bundle['learner'] == 'x':
            result = x_learner(test[FEATURES], model_bundle=model_bundle)
        else:
            raise ValueError('未知的 Learner 类型')
        test_scores[name] = result['cate']
        if name == 't_xgboost':
            propensity = result['mu1']
    test_scores['propensity'] = propensity

    budgets = sorted({*DEFAULT_BUDGETS, selection["primary_budget"]})
    progress("用冻结评分计算 Test 点估计和主指标配对区间")
    evaluation = evaluate_ranking(
        test[outcome], test.treatment, test_scores, assignment_probability,
        budgets=budgets, primary_metric=selection["primary_metric"],
        primary_budget=selection["primary_budget"], n_bootstrap=selection["n_bootstrap"], seed=seed,
    )
    write_result(evaluation["metrics"], out / 'test_metrics.csv')
    write_result(evaluation["curves"], out / 'curves.csv')
    write_result(evaluation["bootstrap"]["summary"], out / 'test_bootstrap.csv')
    write_result(evaluation["bootstrap"]["pairwise"], out / 'test_pairwise.csv')

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
    return evaluation
