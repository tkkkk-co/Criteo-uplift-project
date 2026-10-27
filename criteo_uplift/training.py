# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。
# Notebook 03 的核心实现：run_learnability_diagnostics → fit_and_select_models。
# 输出 learnability.csv、validation_*.csv、模型文件和冻结选择 selection.json；见 docs/RESULTS_GUIDE.md。

import time

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES
from .pipeline_io import safe_output, write_json
from .diagnostics import learnability
from .evaluation import evaluate_policies, ranking_metrics
from .models import make_model


#-----------------------------------------------------------------------------------------------------------------------
# 函数：run_learnability_diagnostics —— 检查特征对访问和转化的预测信息。
# 输入：
#     train：训练集数据表。
#     validation：验证集数据表，用于评价和策略选择。
#     out：主流程已创建的诊断结果目录。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
# 输出：
#     返回值：None；保存 learnability.csv。
# 说明：分别检查特征对访问和转化的预测能力，在验证集评价并保存两组结果。
#-----------------------------------------------------------------------------------------------------------------------
def run_learnability_diagnostics(train, validation, out, *, seed):
    records = [
        learnability(train, validation, FEATURES, outcome, seed=seed)
        for outcome in ("visit", "conversion")
    ]
    out = safe_output(out)
    pd.concat(records).to_csv((safe_output(out / 'learnability.csv')), index=False)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：fit_and_select_models —— 拟合四个候选模型并冻结验证集策略选择。
# 输入：
#     train：训练集数据表。
#     validation：验证集数据表，用于评价和策略选择。
#     outcome：当前结果列名，通常为 conversion 或 visit。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
#     trees：XGBoost 候选模型使用的树数量。
#     primary_budget：主策略允许触达的人群比例。
#     assignment_probability：固定使用的处理分配概率估计值。
#     out：主流程已创建的模型与验证集结果目录。
#     progress：接收进度文字的函数。
# 输出：字典，各字段如下。
#     models：已拟合模型字典。
#     model_metadata：模型参数和训练耗时。
#     selection：验证集上冻结的策略选择记录。
# 说明：使用验证集主预算收益选择策略，保存模型和冻结记录，再由独立步骤开展测试评价。
#-----------------------------------------------------------------------------------------------------------------------
def fit_and_select_models(
    train,
    validation,
    *,
    outcome,
    seed,
    trees,
    primary_budget,
    assignment_probability,
    out,
    progress=print,
):
    out = safe_output(out)
    models = {}
    validation_scores = {}
    validation_ranking = []
    model_metadata = {}
    model_names = ["s_logistic", "t_logistic", "t_xgboost", "x_xgboost"]

    for name in model_names:
        progress("训练 " + name)
        kwargs = {"n_estimators": trees} if "xgboost" in name else {}
        model = make_model(name, random_state=seed, **kwargs)
        started = time.perf_counter()
        model.fit(train[FEATURES], train.treatment, train[outcome])
        model_metadata[name] = {
            "parameters": model.get_params(),
            "seconds": time.perf_counter() - started,
        }
        validation_scores[name] = model.predict(validation[FEATURES])
        models[name] = model
        validation_ranking.append(
            {
                "policy": name,
                **ranking_metrics(
                    validation[outcome],
                    validation.treatment,
                    validation_scores[name],
                    assignment_probability,
                    seed,
                ),
            }
        )
        joblib.dump(model, (safe_output(out / f'{name}.joblib')))

    # 响应概率基线预测处理状态下的结果概率，区别于处理分配概率。
    validation_scores["propensity"] = models["t_xgboost"].predict_outcomes(
        validation[FEATURES]
    )[1]
    validation_policies = evaluate_policies(
        validation[outcome],
        validation.treatment,
        validation_scores,
        assignment_probability,
        budgets=[primary_budget],
        bootstrap=0,
        seed=seed,
    )
    validation_policies.to_csv((safe_output(out / 'validation_policies.csv')), index=False)
    pd.DataFrame(validation_ranking).to_csv(
        (safe_output(out / 'validation_ranking.csv')), index=False
    )

    # 分别冻结全部可用策略中的最佳策略，以及仅增量模型中的最佳候选。
    selected_policy = (
        validation_policies.sort_values(
            "incremental_per_100k", ascending=False, kind="stable"
        )
        .iloc[0]
        .policy
    )
    selected_uplift_model = (
        validation_policies[validation_policies.policy.isin(model_names)]
        .sort_values("incremental_per_100k", ascending=False, kind="stable")
        .iloc[0]
        .policy
    )
    selection = {
        "primary_budget": primary_budget,
        "criterion": "validation HT incremental responses per 100k eligible people",
        "selected_policy": selected_policy,
        "selected_uplift_model": selected_uplift_model,
        "candidates": ["random", "propensity", *model_names],
        "frozen_before_test": True,
        "note": (
            "All models remain trained on training split only; no test tuning, "
            "no train+validation refit."
        ),
    }
    write_json(selection, (out / 'selection.json'))
    return {
        "models": models,
        "model_metadata": model_metadata,
        "selection": selection,
    }
