# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。
# Notebook 03 的核心实现：fit_and_select_models。
# 输出 validation_*.csv、模型文件和冻结选择 selection.json；见 docs/RESULTS_GUIDE.md。

import joblib
import pandas as pd

from .config import FEATURES
from .pipeline_io import safe_output, write_result
from .evaluation import evaluate_policies
from .metrics import evaluate_ranking
from .models import s_learner, t_learner, x_learner

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
    #-------------
    # 函数名：fit_and_select_models
    # 功能：使用 Train 拟合全部候选 S/T/X 模型，在 Validation 比较排序与策略价值，保存模型并冻结策略选择。
    # 输入参数：
    #     - train：用于模型拟合的训练集 DataFrame。
    #     - validation：仅用于候选模型和策略选择的验证集 DataFrame。
    #     - outcome：本次训练使用的二元结果列名。
    #     - seed：模型及排序随机种子。
    #     - trees：XGBoost 候选模型的树数量。
    #     - primary_budget：用于 Validation 选型的主要触达预算。
    #     - assignment_probability：用于 HT 指标的处理分配概率代入值。
    #     - out：模型、验证指标和 selection.json 的输出目录。
    #     - progress：接收模型训练进度的回调函数。
    # 输出参数：
    #     - 结果字典：包含拟合模型、模型参数/耗时 metadata 和冻结 selection。
    #-------------
    out = safe_output(out)
    # 1. 固定四种候选：逐个直接调用 Learner，Train 拟合、Validation 预测。
    X_train, X_val = train[FEATURES], validation[FEATURES]
    t_train, y_train = train.treatment, train[outcome]
    logistic_params = {'random_state': seed}
    xgb_params = {'random_state': seed, 'n_estimators': trees}
    results = {}

    progress('训练 s_logistic')
    results['s_logistic'] = s_learner(
        X_val, X_train=X_train, t=t_train, y=y_train,
        outcome_model='logistic', outcome_params=logistic_params,
    )
    progress('训练 t_logistic')
    results['t_logistic'] = t_learner(
        X_val, X_train=X_train, t=t_train, y=y_train,
        outcome_model='logistic', outcome_params=logistic_params,
    )
    progress('训练 t_xgboost')
    results['t_xgboost'] = t_learner(
        X_val, X_train=X_train, t=t_train, y=y_train,
        outcome_model='xgboost', outcome_params=xgb_params,
    )
    progress('训练 x_xgboost')
    results['x_xgboost'] = x_learner(
        X_val, X_train=X_train, t=t_train, y=y_train,
        outcome_model='xgboost', effect_model='xgboost',
        outcome_params=xgb_params, effect_params=xgb_params,
    )

    validation_scores = {name: result['cate'] for name, result in results.items()}

    # 2. baseline：传统转化倾向response propenity score：预测 T=1 的 Y=1 概率，与 Treatment Propensity 区分。
    validation_scores['propensity'] = results['t_xgboost']['mu1']
    names = list(results)
    validation_ranking = [
        {
            "policy": name,
            **evaluate_ranking(
                validation[outcome],
                validation.treatment,
                validation_scores[name],
                assignment_probability,
                seed=seed,
                include_curves=False,
            )["metrics"],
        }
        for name in names
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
    selected_uplift_model = ordered_policies[ordered_policies.policy.isin(names)].iloc[0].policy
    selection = {
        "primary_budget": primary_budget,
        "criterion": "validation HT incremental responses per 100k eligible people",
        "selected_policy": selected_policy,
        "selected_uplift_model": selected_uplift_model,
        "candidates": ["random", "propensity", *names],
        "frozen_before_test": True,
        "note": (
            "All models remain trained on training split only; no test tuning, "
            "no train+validation refit."
        ),
    }
    # 把用于报告与 Test 的保存逻辑集中在末尾。
    outcome_support = []
    for arm in (0, 1):
        labels = train.loc[train.treatment == arm, outcome]
        positive, n = int(labels.sum()), len(labels)
        outcome_support.append({
            'treatment': arm, 'n': n, 'positive': positive,
            'negative': n - positive, 'single_class': positive in (0, n),
        })
    models = {name: result['model_bundle'] for name, result in results.items()}
    model_metadata = {
        name: {
            'parameters': result['model_bundle']['parameters'],
            'outcome_support': outcome_support,
            'seconds': result['fit_seconds'],
        }
        for name, result in results.items()
    }
    for name, model_bundle in models.items():
        joblib.dump(model_bundle, safe_output(out / f'{name}.joblib'))

    write_result(selection, (out / 'selection.json'))
    return {
        "models": models,
        "model_metadata": model_metadata,
        "selection": selection,
    }
