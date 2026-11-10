# 使用合成随机实验数据演示训练、策略选择和评价，结果不代表真实 Criteo 实验。

import time
import numpy as np
import pandas as pd
from scipy.special import expit
from .preparation import FEATURES, split_sample
from .pipeline_io import safe_output, write_result
from .models import predict_policy_scores
from .training import fit_and_select_models
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
    parts, _, _ = split_sample(frame, seed=seed)
    train, val, test = (parts[k] for k in ['train', 'validation', 'test'])
    e = float(train.treatment.mean())
    output.mkdir(parents=True)
    for directory in (output / 'models', output / 'evaluation'):
        directory.mkdir()
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
    write_result(protocol, (output / 'protocol.json'))
    # 与主实验共用训练和 validation 冻结逻辑，合成数据身份记录在 protocol 中。
    fitted = fit_and_select_models(
        train,
        val,
        outcome='conversion',
        seed=seed,
        trees=trees,
        primary_budget=0.2,
        assignment_probability=e,
        out=output / 'models',
    )
    scores = predict_policy_scores(fitted['models'], test[FEATURES])
    policies = evaluate_policies(
        test.conversion,
        test.treatment,
        scores,
        e,
        budgets=[0.2],
        bootstrap=bootstrap,
        seed=seed,
    )
    write_result(policies, output / 'evaluation' / 'test_policies.csv')
    write_result(pd.DataFrame(
        [
            {
                'policy': name,
                **ranking_metrics(test.conversion, test.treatment, values, e, seed),
            }
            for name, values in scores.items()
        ]
    ), output / 'evaluation' / 'ranking.csv')
    write_result(
        {
            **protocol,
            'status': 'complete',
            'selection': fitted['selection'],
            'elapsed_seconds': time.perf_counter() - start,
        },
        (output / 'manifest.json'),
    )
    return policies
