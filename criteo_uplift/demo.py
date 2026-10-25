# 使用合成随机实验数据演示训练、策略选择和评价，结果不代表真实 Criteo 实验。

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from .data import FEATURES, split_sample
from .pipeline_io import safe_output, write_json
from .models import make_model
from .evaluation import evaluate_policies, ranking_metrics


#-----------------------------------------------------------------------------------------------------------------------
# 函数：run_demo —— 使用合成数据演示完整建模与评价。
# 输入：
#     output：本次任务的新输出路径。
#     n：样本数量或预期数组长度。
#     seed：控制抽样、同分排序或模型拟合的随机种子。
#     trees：XGBoost 候选模型使用的树数量。
#     bootstrap：重抽样次数；支持时可用 0 仅计算点估计。
# 输出：
#     policies：主预算下的策略评价结果表。
# 说明：生成可复现的合成随机实验，只在验证集选择策略，再保存测试评价和运行说明。
#-----------------------------------------------------------------------------------------------------------------------
def run_demo(output, n=6000, seed=42, trees=20, bootstrap=30):
    output = safe_output(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if not isinstance(n, int) or n < 1000 or not isinstance(trees, int) or trees < 1:
        raise ValueError('n must be >=1000 and trees >=1')
    if not isinstance(bootstrap, int) or bootstrap < 2:
        raise ValueError('bootstrap must be >=2')
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    mu0 = expit(-2.5 + 0.7 * X[:, 0])
    mu1 = np.clip(mu0 + 0.06 * np.tanh(X[:, 1]), 0.001, 0.999)
    treatment = rng.binomial(1, 0.65, n)
    y = rng.binomial(1, np.where(treatment == 1, mu1, mu0))
    frame = pd.DataFrame(X, columns=FEATURES)
    frame['row_id'] = np.arange(n)
    frame['treatment'], frame['conversion'] = treatment, y
    frame['true_tau'] = mu1 - mu0
    parts = split_sample(frame, seed=seed)
    train, val, test = (parts[k] for k in ['train', 'validation', 'test'])
    e = float(train.treatment.mean())
    names = ['s_logistic', 't_logistic', 't_xgboost', 'x_xgboost']
    models, val_scores = {}, {}
    output.mkdir(parents=True)
    protocol = {
        'dataset': 'synthetic randomized demonstration',
        'is_criteo_result': False,
        'n': n,
        'seed': seed,
        'trees': trees,
        'bootstrap': bootstrap,
        'budget': 0.2,
        'DGP': 'Normal X; mu0=expit(-2.5+.7*X0), mu1=clip(mu0+.06*tanh(X1),.001,.999); T~Bernoulli(.65)',
        'purpose': 'Verify installation and exercise the real S/T/X + policy evaluation APIs.',
    }
    write_json(protocol, output / 'protocol.json')
    for name in names:
        model = make_model(
            name,
            random_state=seed,
            **({'n_estimators': trees} if 'xgboost' in name else {}),
        )
        model.fit(train[FEATURES], train.treatment, train.conversion)
        models[name] = model
        val_scores[name] = model.predict(val[FEATURES])
    val_scores['propensity'] = models['t_xgboost'].predict_outcomes(val[FEATURES])[1]
    validation = evaluate_policies(
        val.conversion,
        val.treatment,
        val_scores,
        e,
        budgets=[0.2],
        bootstrap=0,
        seed=seed,
    )
    winner = (
        validation.sort_values('incremental_per_100k', ascending=False, kind='stable')
        .iloc[0]
        .policy
    )
    selection = {
        'selected_policy': winner,
        'frozen_before_test': True,
        'criterion': 'validation 20% HT gain',
        'synthetic': True,
    }
    write_json(selection, output / 'selection.json')
    validation.to_csv(output / 'validation_policies.csv', index=False)
    scores = {name: model.predict(test[FEATURES]) for name, model in models.items()}
    scores['propensity'] = models['t_xgboost'].predict_outcomes(test[FEATURES])[1]
    policies = evaluate_policies(
        test.conversion,
        test.treatment,
        scores,
        e,
        budgets=[0.2],
        bootstrap=bootstrap,
        seed=seed,
    )
    policies.to_csv(output / 'test_policies.csv', index=False)
    pd.DataFrame(
        [
            {
                'policy': name,
                **ranking_metrics(test.conversion, test.treatment, values, e, seed),
            }
            for name, values in scores.items()
        ]
    ).to_csv(output / 'ranking.csv', index=False)
    write_json(
        {
            **protocol,
            'status': 'complete',
            'selection': selection,
            'elapsed_seconds': time.perf_counter() - start,
        },
        output / 'manifest.json',
    )
    return policies
