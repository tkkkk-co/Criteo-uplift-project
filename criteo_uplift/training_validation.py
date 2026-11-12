# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。
# Notebook 03 的核心实现：fit_and_select_models。
# 输出 validation_*.csv、模型文件和冻结选择 selection.json；见 docs/RESULTS_GUIDE.md。

import joblib
import pandas as pd

from .config import FEATURES
from .pipeline_io import safe_output, write_result
from .evaluation import evaluate_policies
from .metrics import ranking_metrics
from .models import LEARNERS
from .diagnostics import diagnose_overlap

MODEL_CANDIDATES = [
    {"name": "s_logistic", "learner": "s", "outcome_model": "logistic"},
    {"name": "t_logistic", "learner": "t", "outcome_model": "logistic"},
    {"name": "t_xgboost", "learner": "t", "outcome_model": "xgboost"},
    {"name": "x_xgboost", "learner": "x", "outcome_model": "xgboost", "effect_model": "xgboost"},
]


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
    models = {}
    validation_scores = {}
    model_metadata = {}

    # 在任何模型拟合之前，仅对当前 Train 执行一次经验覆盖诊断。
    coverage = diagnose_overlap(train[FEATURES], train.treatment)
    diagnostics_out = out.parent / 'diagnostics'
    write_result(coverage['summary'], diagnostics_out / 'overlap_positivity.json')
    write_result(coverage['strata'], diagnostics_out / 'overlap_strata.csv')

    names = [candidate["name"] for candidate in MODEL_CANDIDATES]
    for candidate in MODEL_CANDIDATES:
        name = candidate["name"]
        progress("训练 " + name)
        kwargs = {"n_estimators": trees} if "xgboost" in name else {}
        result = LEARNERS[candidate["learner"]](
            validation[FEATURES], X_train=train[FEATURES], t=train.treatment, y=train[outcome],
            **{k: v for k, v in candidate.items() if k not in {"name", "learner"}},
            random_state=seed, overlap=coverage['summary'], **kwargs,
        )
        model = result['model']
        validation_scores[name] = result['cate']
        if name == 't_xgboost':
            propensity = result['mu1']
        model_metadata[name] = {
            "parameters": model['parameters'],
            "overlap_status": model['overlap']['status'],
            "overlap_input_signature": model['overlap']['input_signature'],
            "overlap_file": "diagnostics/overlap_positivity.json",
            "outcome_support": model['outcome_support'],
            "seconds": result["fit_seconds"],
        }
        models[name] = model
        joblib.dump(model, safe_output(out / f'{name}.joblib'))

    validation_scores['propensity'] = propensity
    validation_ranking = [
        {
            "policy": name,
            **ranking_metrics(
                validation[outcome], validation.treatment, validation_scores[name],
                assignment_probability, seed,
            ),
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
    write_result(selection, (out / 'selection.json'))
    return {
        "models": models,
        "model_metadata": model_metadata,
        "overlap": coverage["summary"],
        "selection": selection,
    }
