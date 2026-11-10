# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。
# Notebook 03 的核心实现：fit_and_select_models。
# 输出 validation_*.csv、模型文件和冻结选择 selection.json；见 docs/RESULTS_GUIDE.md。

import time

import joblib
import pandas as pd

from .config import FEATURES
from .pipeline_io import safe_output, write_result
from .evaluation import evaluate_policies
from .metrics import ranking_metrics
from .models import make_uplift_model, predict_policy_scores

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
    out = safe_output(out)
    models = {}
    model_metadata = {}

    names = [candidate["name"] for candidate in MODEL_CANDIDATES]
    for candidate in MODEL_CANDIDATES:
        name = candidate["name"]
        progress("训练 " + name)
        kwargs = {"n_estimators": trees} if "xgboost" in name else {}
        model = make_uplift_model(**{k: v for k, v in candidate.items() if k != "name"},
                                  random_state=seed, **kwargs)
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
        "selection": selection,
    }
