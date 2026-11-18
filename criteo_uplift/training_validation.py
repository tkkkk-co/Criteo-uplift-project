# 在训练集拟合模型，在验证集选择策略；不使用测试结果进行选择。

import joblib
import warnings
from datetime import datetime, timezone
import numpy as np

from .config import FEATURES, POLICY_NAMES, UPLIFT_NAMES
from .pipeline_io import safe_output, write_result
from .metrics import evaluate_ranking
from .models import s_learner, t_learner, x_learner

def _fit_candidates(train, validation, outcome, seed, trees, progress):
    """Six CATE estimators and two independently named treated-response policies."""
    X_train, X_val = train[FEATURES], validation[FEATURES]
    results = {}
    for learner in ('s', 't', 'x'):
        for model in ('logistic', 'xgboost'):
            name = f'{learner}_{model}'
            progress(f'训练 {name}')
            params = {'random_state': seed, **({'n_estimators': trees} if model == 'xgboost' else {})}
            kwargs = dict(X_train=X_train, t=train.treatment, y=train[outcome],
                          outcome_model=model, outcome_params=params)
            if learner == 'x':
                kwargs.update(effect_model='linear' if model == 'logistic' else 'xgboost',
                              effect_params=params if model == 'xgboost' else {})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                results[name] = {'s': s_learner, 't': t_learner, 'x': x_learner}[learner](X_val, **kwargs)
            results[name]['warnings'] = [str(item.message) for item in caught]
            for message in results[name]['warnings']:
                progress(f'{name} 训练警告：{message}')
    for model in ('logistic', 'xgboost'):
        source = results[f't_{model}']
        bundle = source['model_bundle']
        results[f'response_{model}'] = dict(
            model_bundle={**bundle, 'learner': 'response', 'fit_source': f't_{model}',
                          'mu1_fitted_model': bundle['mu1_fitted_model']},
            score=source['mu1'], fit_seconds=0.0,
        )
        # The serialized response artifact contains only the treated outcome model.
        results[f'response_{model}']['model_bundle'].pop('mu0_fitted_model')
    return results


def policy_scores(results):
    return {name: result['score'] if name.startswith('response_') else result['cate']
            for name, result in results.items()}


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
    #     - n_bootstrap：冻结后 Test 评价使用的重抽样次数，Validation 仅计算点估计。
    #     - assignment_probability：用于 HT 指标的处理分配概率代入值。
    #     - out：模型、验证指标和 selection.json 的输出目录。
    #     - progress：接收模型训练进度的回调函数。
    # 输出参数：
    #     - 结果字典：模型参数、拟合耗时和冻结选择。
    #-------------
    out = safe_output(out)
    if n_bootstrap < 2:
        raise ValueError("Test 评价需要至少两次 Bootstrap")
    results = _fit_candidates(train, validation, outcome, seed, trees, progress)
    validation_scores = policy_scores(results)

    # Response 已复用对应 T-Learner 处理组模型。八种评分共同评价。Validation 只评价预设覆盖率，面积仍使用完整曲线。
    evaluation = evaluate_ranking(
        validation[outcome], validation.treatment, validation_scores, assignment_probability,
        include_random=True, budgets=[primary_budget], primary_metric=primary_metric, primary_budget=primary_budget,
        n_bootstrap=0, seed=seed, include_curves=False,
    )

    # 4. 按主指标点估计排序。随机参考线不参与选择，同分沿用固定候选顺序。
    ordered = evaluation["metrics"].query("policy != 'random'").sort_values(
        primary_metric, ascending=False, kind="stable", na_position="last",
    )
    uplift = ordered[ordered.policy.isin(UPLIFT_NAMES)]
    if ordered[primary_metric].isna().all() or uplift[primary_metric].isna().all():
        raise ValueError("Validation 主指标没有可用的点估计，无法冻结策略或 uplift 候选")
    selection = {
        "primary_metric": primary_metric,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_probability": assignment_probability,
        "seed": seed,
        "primary_budget": primary_budget,
        "selection_statistic": "point_estimate",
        "validation_n_bootstrap": 0,
        "n_bootstrap": n_bootstrap,
        "selected_policy": ordered.iloc[0].policy,
        "selected_uplift_model": uplift.iloc[0].policy,
        "candidates": POLICY_NAMES,
        "frozen_before_test": True,
        "note": "Models are fitted on Train only. Validation selects by the preset metric point estimate. Bootstrap is used for Test inference.",
    }

    # 5. 保存 Validation 评价、冻结选择和 Train 模型，供 Test 独立预测。
    write_result(evaluation["metrics"], out / 'validation_metrics.csv')
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
            'fit_source': result['model_bundle'].get('fit_source', name),
            'training_status': 'fitted',
            'warnings': result.get('warnings', []),
            'validation_score_std': float(np.std(validation_scores[name])),
            'components': _model_diagnostics(result['model_bundle']),
        }
        for name, result in results.items()
    }
    for name, model_bundle in models.items():
        joblib.dump(model_bundle, safe_output(out / f'{name}.joblib'))

    np.savez_compressed(out / 'validation_predictions.npz', row_id=validation.row_id.to_numpy(),
                        y=validation[outcome].to_numpy(), t=validation.treatment.to_numpy(), **validation_scores)
    write_result(model_metadata, out / 'training_metadata.json')
    write_result(selection, (out / 'selection.json'))
    return {
        "model_metadata": model_metadata,
        "selection": selection,
    }


def _model_diagnostics(bundle):
    """Record convergence and splits from fitted objects, without tuning on Test."""
    diagnostics = {}
    for key, model in bundle.items():
        if not key.endswith('model'):
            continue
        fitted = model.steps[-1][1] if hasattr(model, 'steps') else model
        item = {'type': type(fitted).__name__}
        if hasattr(fitted, 'n_iter_'):
            item['iterations'] = np.asarray(fitted.n_iter_).tolist()
            item['iteration_limit_reached'] = bool(np.max(fitted.n_iter_) >= fitted.max_iter)
        if hasattr(fitted, 'get_booster'):
            tree = fitted.get_booster().trees_to_dataframe()
            item['trees'] = int(tree.Tree.nunique())
            item['split_nodes'] = int(tree.Feature.ne('Leaf').sum())
        diagnostics[key] = item
    return diagnostics
