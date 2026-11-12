#-------------
# 独立分析固定 C2ST 样本；只读诊断缓存，不读取原始 CSV，也不修改正式结果。
#-------------
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from criteo_uplift.config import FEATURES, configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import skew
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from threadpoolctl import threadpool_limits

from criteo_uplift.diagnostics import _features, _make_c2st_model, covariate_balance
from criteo_uplift.pipeline_io import safe_output
from criteo_uplift.preparation import _load_diagnostics_cache


def load_sample(run, cache):
    #-------------
    # 校验缓存来源、精度、完整性和有序行号哈希，禁止自动重新抽样。
    # 输入参数：
    #     - run：包含正式 diagnostics 结果的实验目录。
    #     - cache：已有的诊断 NPZ 文件路径。
    # 输出参数：
    #     - 样本 DataFrame 和正式诊断详情字典。
    #-------------
    details = json.loads((run / 'diagnostics/c2st_details.json').read_text())
    result = _load_diagnostics_cache(cache, details['source'], details['sample_n'], details['seed'])
    if result is None:
        raise ValueError('诊断缓存缺失或校验失败；请提供原有有效缓存，本脚本不会读取原始数据')
    sample = result['c2st_sample']
    digest = hashlib.sha256(sample.row_id.to_numpy().tobytes()).hexdigest()
    if digest != details['sample_row_ids_sha256']:
        raise ValueError('诊断样本有序行号哈希与正式结果不一致')
    if details['sample_n'] != 20000 or details['permutations'] != 499:
        raise ValueError('本次调查要求固定的 20000 行样本与 499 次置换')
    return sample, details


def make_model(analysis, seed, bounds):
    #-------------
    # 复用正式 Logistic 工厂，只替换指定的预处理步骤。
    # 输入参数：
    #     - analysis：baseline、robust 或 winsor。
    #     - seed：与正式诊断一致的模型种子。
    #     - bounds：仅由训练特征确定的 1% 和 99% 缩尾阈值。
    # 输出参数：
    #     - 未拟合的 Logistic Pipeline。
    #-------------
    model = _make_c2st_model('logistic', seed)
    if analysis == 'baseline':
        return model
    if analysis == 'robust':
        model.steps[0] = ('robustscaler', RobustScaler())
        return model
    if analysis == 'winsor':
        clip = FunctionTransformer(np.clip, kw_args={'a_min': bounds[0], 'a_max': bounds[1]})
        return make_pipeline(clip, model)
    raise ValueError(f'未知分析名称：{analysis}')


def tail_table(x, treatment):
    #-------------
    # 按处理臂描述全部诊断样本的分布；尾部比例共用合并样本阈值。
    # 输入参数：
    #     - x：20000 行、12 列的原始 float64 特征矩阵。
    #     - treatment：与特征逐行对应的二元处理标签。
    # 输出参数：
    #     - 每个特征、每个处理臂一行的描述性统计表。
    #-------------
    pooled = np.quantile(x, [.01, .99], axis=0)
    records = []
    for j, feature in enumerate(FEATURES):
        for arm in (0, 1):
            values = x[treatment == arm, j]
            q = np.quantile(values, [.01, .05, .25, .5, .75, .95, .99, .999])
            records.append(dict(
                feature=feature, arm='Treatment' if arm else 'Control', n=len(values),
                minimum=values.min(), maximum=values.max(), p1=q[0], p5=q[1],
                p50=q[3], p95=q[5], p99=q[6], p999=q[7], iqr=q[4]-q[2],
                skewness=skew(values, bias=False), pooled_p1=pooled[0, j],
                pooled_p99=pooled[1, j], below_pooled_p1=np.mean(values < pooled[0, j]),
                above_pooled_p99=np.mean(values > pooled[1, j]),
            ))
    return pd.DataFrame(records)


def analyze(run, cache, output, winsorize=False):
    #-------------
    # 复现基线及完整零分布后，完成系数、标准化和尾部分布的定向分析。
    # 输入参数：
    #     - run：正式实验结果目录，仅读取。
    #     - cache：正式诊断样本缓存，仅读取。
    #     - output：独立的新输出目录，拒绝覆盖已有结果。
    #     - winsorize：描述性诊断支持后才显式启用一次固定 1%/99% 缩尾。
    # 输出参数：
    #     - 无；写出聚合 CSV 与可复核的分析说明 JSON，不导出逐行数据。
    #-------------
    output = safe_output(output)
    if output.exists():
        raise ValueError('分析输出目录已存在，请使用新的独立目录')
    sample, details = load_sample(run, cache)
    seed, permutations = details['seed'], details['permutations']
    # 与正式 c2st 完全相同：即使使用全部样本，也先执行 choice 改变行序并推进随机流。
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(sample), size=details['sample_n'], replace=False)
    ordered = sample.iloc[positions]
    x = _features(ordered, FEATURES)
    t = ordered.treatment.to_numpy(dtype=np.int8)
    train, test = train_test_split(np.arange(len(t)), test_size=.3, stratify=t, random_state=seed)
    x_train, x_test, t_train, t_test = x[train], x[test], t[train], t[test]
    if (len(train), len(test), int(t_train.sum()), int(t_test.sum())) != (14000, 6000, 7000, 3000):
        raise ValueError('诊断划分人数与正式结果不一致')
    # 阈值仅依赖训练特征，和标签无关；同一阈值用于两臂及全部置换。
    bounds = np.quantile(x_train, [.01, .99], axis=0)
    analyses = ['baseline', 'robust'] + (['winsor'] if winsorize else [])
    observed, fitted, null_records = {}, {}, []
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for name in analyses:
            model = make_model(name, seed, bounds)
            model.fit(x_train, t_train)
            fitted[name] = model
            observed[name] = float(roc_auc_score(t_test, model.predict_proba(x_test)[:, 1]))
        # 对照 diagnostics.c2st：分别打乱原始训练/留出标签，每轮创建并完整拟合新 Pipeline。
        # 各预处理版本共用同一轮标签；模型不消费此随机流，省略 XGBoost 不改变置换序列。
        for b in range(permutations):
            shuffled_train = rng.permutation(t_train)
            shuffled_test = rng.permutation(t_test)
            for name in ['baseline', 'robust']:
                model = make_model(name, seed, bounds)
                model.fit(x_train, shuffled_train)
                auc = roc_auc_score(shuffled_test, model.predict_proba(x_test)[:, 1])
                null_records.append(dict(analysis=name, permutation=b+1, roc_auc=float(auc)))
    null = pd.DataFrame(null_records)
    reference = pd.read_csv(run / 'diagnostics/c2st.csv').set_index('model').loc['logistic']
    reference_null = pd.read_csv(run / 'diagnostics/c2st_null.csv')
    reference_null = reference_null.loc[reference_null.model.eq('logistic')].sort_values('permutation')
    baseline_null = null.loc[null.analysis.eq('baseline'), 'roc_auc'].to_numpy()
    np.testing.assert_allclose(observed['baseline'], reference.roc_auc, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(baseline_null, reference_null.roc_auc, rtol=1e-12, atol=1e-15)
    summaries = []
    for name in analyses:
        # 缩尾由分组尾部描述触发，仅报告效果量，避免将事后选择当成预先指定的显著性检验。
        if name == 'winsor':
            summaries.append(dict(analysis=name, holdout_auc=observed[name],
                                  delta_auc=observed[name]-observed['baseline'],
                                  pvalue=None, exceedances=None, permutations=0,
                                  null_mean=None, null_sd=None))
            continue
        values = null.loc[null.analysis.eq(name), 'roc_auc'].to_numpy()
        exceedances = int(np.count_nonzero(values >= observed[name]))
        summaries.append(dict(analysis=name, holdout_auc=observed[name],
                              delta_auc=observed[name]-observed['baseline'],
                              pvalue=(1+exceedances)/(permutations+1),
                              exceedances=exceedances, permutations=permutations,
                              null_mean=values.mean(), null_sd=values.std(ddof=1)))
    if summaries[0]['pvalue'] != reference.pvalue:
        raise ValueError('基线置换 p 值未复现，停止输出敏感性结论')
    coef = fitted['baseline'][-1].coef_[0]
    robust_coef = fitted['robust'][-1].coef_[0]
    balance = pd.read_csv(run / 'diagnostics/balance.csv').set_index('feature').loc[FEATURES]
    sample_balance = covariate_balance(ordered, FEATURES).set_index('feature').loc[FEATURES]
    coefficients = pd.DataFrame(dict(
        feature=FEATURES, SMD=balance.smd.to_numpy(), logistic_coef=coef, abs_coef=np.abs(coef),
        coef_sign=np.sign(coef).astype(int), smd_sign=np.sign(balance.smd).astype(int).to_numpy(),
        beta_times_smd=coef*balance.smd.to_numpy(),
        direction_alignment=np.sign(coef)==np.sign(balance.smd.to_numpy()),
        sample_smd=sample_balance.smd.to_numpy(), robust_coef=robust_coef,
        robust_sign_changed=np.sign(coef)!=np.sign(robust_coef),
    ))
    corr = np.corrcoef(x_train, rowvar=False)
    correlations = pd.DataFrame([
        dict(feature_a=FEATURES[i], feature_b=FEATURES[j], pearson_r=corr[i, j])
        for i in range(len(FEATURES)) for j in range(i+1, len(FEATURES))
    ]).sort_values('pearson_r', key=np.abs, ascending=False).head(3)
    output.mkdir(parents=True)
    for name, frame in [('summary', pd.DataFrame(summaries)), ('coefficients', coefficients),
                        ('tails', tail_table(x, t)), ('null_auc', null), ('correlations', correlations)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    if winsorize:
        pd.DataFrame(dict(feature=FEATURES, lower=bounds[0], upper=bounds[1])).to_csv(
            output / 'winsor_thresholds.csv', index=False)
    metadata = dict(
        scope='exploratory sensitivity; separate from prespecified two-model Holm family',
        seed=seed, sample_n=len(t), train_n=len(train), test_n=len(test),
        sample_row_ids_sha256=details['sample_row_ids_sha256'],
        train_row_ids_sha256=hashlib.sha256(ordered.row_id.to_numpy()[train].tobytes()).hexdigest(),
        test_row_ids_sha256=hashlib.sha256(ordered.row_id.to_numpy()[test].tobytes()).hexdigest(),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        baseline_auc_and_all_499_null_values_verified=True,
        logistic_parameters=fitted['baseline'][-1].get_params(),
        robust_scaler_parameters=fitted['robust'][0].get_params(),
        full_source_smd_used=True, sample_smd_also_exported=True,
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        raw_csv_read=False, winsorization=bool(winsorize),
        winsor_pvalue_not_computed_reason='Selected after descriptive arm-tail comparisons; effect size only',
        code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [Path(__file__), ROOT/'criteo_uplift/diagnostics.py']},
    )
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


def main():
    #-------------
    # 解析独立分析所需的现有文件路径；不提供重新扫描或运行主流程的入口。
    # 输入参数：
    #     - 无函数参数；命令行接收 run、cache、output 及可选 winsorize。
    # 输出参数：
    #     - 无；调用分析函数生成聚合结果。
    #-------------
    parser = argparse.ArgumentParser(description='固定 C2ST 样本的定向敏感性分析')
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--winsorize', action='store_true', help='描述性证据支持后启用一次 1%/99% 缩尾')
    args = parser.parse_args()
    analyze(args.run, args.cache, args.output, args.winsorize)


if __name__ == '__main__':
    main()
