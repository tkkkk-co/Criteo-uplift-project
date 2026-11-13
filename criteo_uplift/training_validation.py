# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。
# Notebook 03 的核心实现：fit_and_select_models。
# 输出 validation_*.csv、模型文件和冻结选择 selection.json；见 docs/RESULTS_GUIDE.md。

import joblib

from .config import FEATURES
from .pipeline_io import safe_output, write_result
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
    primary_metric="gain_ht",
    n_bootstrap=200,
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
    #     - primary_budget：Gain 主指标的选型覆盖率。
    #     - primary_metric：选型使用的唯一主指标。
    #     - n_bootstrap：选型的分层配对重抽样次数。
    #     - assignment_probability：用于 HT 指标的处理分配概率代入值。
    #     - out：模型、验证指标和 selection.json 的输出目录。
    #     - progress：接收模型训练进度的回调函数。
    # 输出参数：
    #     - 结果字典：模型参数、拟合耗时和冻结选择。
    #-------------
    out = safe_output(out)
    if n_bootstrap == 0:
        raise ValueError("Validation 选型需要至少两次 Bootstrap")
    # 1. 训练模型：仅用 Train 拟合四个固定候选，同时预测 Validation 的 CATE。
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

    # 2. 构建基线：复用 T-XGBoost 的 mu1，按接受处理时的转化概率排序。
    validation_scores['propensity'] = results['t_xgboost']['mu1']

    # 3. 一次评价五种评分。Validation 只评价预设覆盖率，面积仍使用完整曲线。
    evaluation = evaluate_ranking(
        validation[outcome], validation.treatment, validation_scores, assignment_probability,
        budgets=[primary_budget], primary_metric=primary_metric, primary_budget=primary_budget,
        n_bootstrap=n_bootstrap, seed=seed, include_curves=False,
    )
    inference = evaluation["bootstrap"]

    # 4. 按主指标的 Bootstrap 均值排序。随机参考线不参与选择，同分沿用固定候选顺序。
    ordered = inference["summary"].query("policy != 'random'").sort_values(
        "bootstrap_mean", ascending=False, kind="stable", na_position="last",
    )
    uplift = ordered[ordered.policy.isin(results)]
    if ordered.bootstrap_mean.isna().all() or uplift.bootstrap_mean.isna().all():
        raise ValueError("Validation 主指标没有可用的 Bootstrap 均值，无法冻结策略或 uplift 候选")
    selection = {
        "primary_metric": primary_metric,
        "primary_budget": primary_budget,
        "selection_statistic": "bootstrap_mean",
        "n_bootstrap": n_bootstrap,
        "selected_policy": ordered.iloc[0].policy,
        "selected_uplift_model": uplift.iloc[0].policy,
        "candidates": list(validation_scores),
        "frozen_before_test": True,
        "note": "Models are fitted on Train only. Validation selects by the preset bootstrap mean.",
    }

    # 5. 保存 Validation 评价、冻结选择和 Train 模型，供 Test 独立预测。
    write_result(evaluation["metrics"], out / 'validation_metrics.csv')
    write_result(inference["summary"], out / 'validation_bootstrap.csv')
    write_result(inference["pairwise"], out / 'validation_pairwise.csv')
    write_result(inference["selection_frequency"], out / 'validation_selection_frequency.csv')
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
        "model_metadata": model_metadata,
        "selection": selection,
    }
