# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。
# Notebook 03 的核心实现：fit_and_select_models。
# 输出 validation_*.csv、模型文件和冻结选择 selection.json；见 docs/RESULTS_GUIDE.md。

import time

import joblib
import pandas as pd

from .preparation import FEATURES
from .pipeline_io import safe_output, write_result
from .evaluation import evaluate_policies, ranking_metrics
from .models import MODEL_NAMES, make_model, predict_policy_scores


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
    model_metadata = {}

    for name in MODEL_NAMES:
        progress("训练 " + name)
        kwargs = {"n_estimators": trees} if "xgboost" in name else {}
        model = make_model(name, random_state=seed, **kwargs)
        started = time.perf_counter()
        model.fit(train[FEATURES], train.treatment, train[outcome])
        model_metadata[name] = {
            "parameters": model.get_params(),
            "seconds": time.perf_counter() - started,
        }
        models[name] = model
        joblib.dump(model, safe_output(out / f'{name}.joblib'))

    validation_scores = predict_policy_scores(models, validation[FEATURES])
    validation_ranking = [
        {
            "policy": name,
            **ranking_metrics(
                validation[outcome], validation.treatment, validation_scores[name],
                assignment_probability, seed,
            ),
        }
        for name in MODEL_NAMES
    ]
    validation_policies = evaluate_policies(
        validation[outcome],
        validation.treatment,
        validation_scores,
        assignment_probability,
        budgets=[primary_budget],
        bootstrap=0,
        seed=seed,
    )
    write_result(validation_policies, out / 'validation_policies.csv')
    write_result(pd.DataFrame(validation_ranking), out / 'validation_ranking.csv')

    # 分别冻结全部可用策略中的最佳策略，以及仅增量模型中的最佳候选。
    ordered_policies = validation_policies.sort_values(
        "incremental_per_100k", ascending=False, kind="stable"
    )
    selected_policy = ordered_policies.iloc[0].policy
    selected_uplift_model = ordered_policies[ordered_policies.policy.isin(MODEL_NAMES)].iloc[0].policy
    selection = {
        "primary_budget": primary_budget,
        "criterion": "validation HT incremental responses per 100k eligible people",
        "selected_policy": selected_policy,
        "selected_uplift_model": selected_uplift_model,
        "candidates": ["random", "propensity", *MODEL_NAMES],
        "frozen_before_test": True,
        "note": (
            "All models remain trained on training split only; no test tuning, "
            "no train+validation refit."
        ),
    }
    write_result(selection, (out / 'selection.json'))
    return {
        "models": models,
        "model_metadata": model_metadata,
        "selection": selection,
    }
