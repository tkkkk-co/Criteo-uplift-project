#!/usr/bin/env python3
"""运行：python run_pipeline.py --sample-size 1000000 --outcome conversion

输出仅 artifacts/，data/ 永远只读。--sample-size 0 为全量建模（较耗内存）。
默认复用校验过来源 SHA256 的随机样本；--diagnostics-only 在现有目录原位更新诊断。
"""

# ============================================================================
# 运行环境与依赖
# 先配置项目内的线程数、OpenMP 等环境，再导入计算库；保留这一执行顺序。
# ============================================================================
from criteo_uplift.runtime import configure_runtime

configure_runtime()

import argparse
from pathlib import Path
import json
import time
import sys
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from criteo_uplift.data import (
    ROOT,
    DEFAULT_DATA,
    FEATURES,
    save_sample,
    load_sample,
    split_sample,
    safe_output,
    write_json,
    fingerprint,
)
from criteo_uplift.models import make_model
from criteo_uplift.diagnostics import (
    scan_raw_data,
    update_experiment_diagnostics,
    learnability,
)
from criteo_uplift.evaluation import (
    ate_summary,
    ranking_metrics,
    uplift_curve,
    evaluate_policies,
    calibration_table,
    ranking_confidence_intervals,
)


# ============================================================================
# 辅助函数：日志、全量 ATE、样本准备
# run() 按分析顺序调用这些函数；它们的定义本身不会启动完整实验。
# ============================================================================
def log(message):
    # 打印时间和进度；flush=True 让长任务运行时及时显示消息。
    print(
        time.strftime('%H:%M:%S'),
        message,
        flush=True,
    )


# -----------------ATE+统计分析-------------------------------
def aggregate_ate(audit, outcome):
    """全量审计只保存聚合量，按二项数据精确重建均值方差。"""
    from scipy.stats import norm

    # 1. 从全量审计中取出 T=0/1 的样本数 n 和正例数 r，无需再加载逐行数据。
    # outcome 是二元结果的列名；主流程在这里分别传入 conversion 和 visit。
    n0, n1 = (
        audit['groups'][str(a)]['n']
        for a in [0, 1]
    )
    r0, r1 = (
        audit['groups'][str(a)][outcome]['positive']
        for a in [0, 1]
    )

    # 2. 两组结果率之差是点估计。这里只比较 treatment 分配，不按 exposure 分组。
    p0, p1 = r0 / n0, r1 / n1
    delta = p1 - p0

    # 3. CI 使用非合并的样本方差估计：对二元 Y，s²/n = p(1-p)/(n-1)。
    # se 是标准误的估计值；取平方根后不能直接称为“无偏标准误”。
    se = np.sqrt(
        p1 * (1 - p1) / (n1 - 1)
        + p0 * (1 - p0) / (n0 - 1)
    )

    # 4. 双侧零差异检验在 H0: p1=p0 下使用合并结果率，区别于上面的 CI。
    pooled = (r0 + r1) / (n0 + n1)
    nullse = np.sqrt(
        pooled * (1 - pooled) * (1 / n0 + 1 / n1)
    )

    # 5. 返回一条结果记录。ATE、SE、CI 保留比例差单位，展示为百分点时再乘 100。
    # relative_uplift 是相对提升；这些统计量不能直接还原真实线上广告收益。
    return {
        'outcome': outcome,
        'n': n0 + n1,
        'n_control': n0,
        'n_treatment': n1,
        'response_control': p0,
        'response_treatment': p1,
        'ate': delta,
        'standard_error': se,
        'ci_low': delta - 1.95996398454 * se,
        'ci_high': delta + 1.95996398454 * se,
        'p_value': (
            2 * norm.sf(
                abs(delta / nullse)
            )
            if nullse > 0
            else 1.0
        ),
        'relative_uplift': delta / p0 if p0 else None,
    }

#----------------------------------------------------------------------------
# 为后续分析准备一份【样本数据】和【全量审计】结果。
# 能安全复用已有抽样缓存时，就加载缓存；否则扫描原始数据并抽样。
# return （sample: DataFrame）, （audit: dict）, （source: dict）, （scanned: dict）
# scanned 保存同一次扫描产生的诊断数据，后面的诊断步骤直接复用。

def prepare_sample(args):
    #args 保留运行参数:(参考 run_pipeline.py line743)
    # - args.data:data的绝对路径
    # - args.sample_size:抽样规模，0=全量
    # - args.seed
    # - args.outcome:选择conversion或visit
    # - args.primary_budget:主策略最多触达的人群比例，默认20%
    # - args.c2st_sample:C2ST 从原始全量来源两组等量抽取的诊断样本规模
    # - args.permutations:C2ST 诊断的置换次数
    # - args.bootstrap:测试集策略评价的 bootstrap 次数
    # - args.trees:XGBoost 模型的树数量
    # - args.output（输出目录）

    if args.c2st_sample < 20 or args.c2st_sample % 20:
        raise ValueError('C2ST sample_size must be >=20 and divisible by 20')
    if args.permutations < 1:
        raise ValueError('permutations must be positive')

    # 1. 定位共享抽样缓存及其元数据；缓存与某次实验的 out 结果目录分开管理。
    base = safe_output(ROOT / 'artifacts')
    base.mkdir(exist_ok=True)
    cache = base / f'cache/sample_{args.sample_size}_seed{args.seed}.npz' # 缓存抽好的样本 下次直接调用不用再抽了
    metadata = cache.with_suffix('.json') # 缓存说明文件

    # 2. 核对原始文件内容、抽样规模和种子后才复用缓存。
    # fingerprint 仍会读取文件计算 SHA256；复用缓存不代表完全不读取原始文件。
    current = fingerprint(args.data)
    reuse = False
    if cache.exists() and metadata.exists():
        info = json.loads(
            metadata.read_text()
        )
        reuse = (
            info['source']['sha256'] == current['sha256']
            and info['sample_size'] == args.sample_size
            and info['seed'] == args.seed
        )
    # 3. 兼容早期已生成的 100 万行、seed=42 缓存：用旧的来源指纹验证。
    # 这是本项目的历史缓存兼容分支，不是通用抽样步骤。
    initial = base / 'source_fingerprint.json'
    if (
        not metadata.exists()
        and cache.exists()
        and initial.exists()
        and args.sample_size == 1_000_000
        and args.seed == 42
    ):
        reuse = (
            json.loads(initial.read_text())['sha256']
            == current['sha256']
        )

    # 4. 只进行一次 CSV 分块扫描，同时准备全量平衡、SRM 和 C2ST 样本。
    # 有建模缓存时跳过建模抽样及完整审计；没有缓存时在同一次扫描中一并完成。
    log('统一扫描：全量检查 + C2ST 抽样' + ('（复用建模缓存）' if reuse else ' + 建模抽样与审计'))
    scanned = scan_raw_data(
        args.data,
        sample_size=None if reuse else args.sample_size,
        seed=args.seed,
        per_arm=args.c2st_sample // 2,
        audit_enabled=not reuse,
        diagnostics_enabled=True,
        progress=log,
        source=current,
    )
    if reuse:
        log('复用已校验来源 SHA256 的随机样本')
        sample = load_sample(cache)
        audit = json.loads(
            (
                metadata
                if metadata.exists()
                else base / 'full_audit.json'
            ).read_text()
        )
        if 'audit' in audit:
            audit = audit['audit']
        for key in ('n_treatment', 'n_control'):
            if audit[key] != scanned['metadata'][key]:
                raise ValueError(f'建模缓存审计与当前全量扫描不一致：{key}')
    else:
        sample, audit = scanned['sample'], scanned['audit']
        save_sample(sample, cache)

    # 建模样本由 sample 单独交给划分步骤，诊断结果无需继续持有这份大表。
    scanned.pop('sample')
    scanned.pop('audit')

    # 5. 保存缓存来源，并返回样本、审计、文件指纹和可直接复用的诊断数据。
    write_json(
        {
            'source': current,
            'sample_size': args.sample_size,
            'seed': args.seed,
            'audit': audit,
        },
        metadata,
    )
    return sample, audit, current, scanned


# ============================================================================
# 实验主流程
# 阅读顺序：数据准备 → 划分与诊断 → 模型拟合 → 验证集选择 → 测试集评价 → 报告。
# out 是本次实验的写入目录；Notebook 中的 RUN 用于定位并读取已保存的结果。
# ============================================================================
def run(args):
    # ------------------------------------------------------------------------
    # 01. 创建本次实验目录：默认按 outcome / sample_size / seed 命名。
    # --output 可指定其他目录；safe_output 禁止写入受保护的 data/。
    # 已有非空目录会报错，防止把不同运行的文件混在一起。
    # ------------------------------------------------------------------------
    start = time.perf_counter()
    out = safe_output(
        args.output
        or ROOT / f'artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}'
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f'结果目录已有正式实验：{out}。仅更新诊断请使用 --diagnostics-only。'
        )
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # 02. 准备样本，再按 treatment × outcome 分层划分为 60% / 20% / 20%。
    # sample 是 DataFrame；audit 描述全量原始数据，不只是抽样后的数据。
    # parts 是字典，键为 train / validation / test，值均为 DataFrame。
    # ------------------------------------------------------------------------
    sample, audit, source, scanned = prepare_sample(args)

    parts = split_sample(
        sample,
        args.outcome,
        args.seed,
    )

    # 后续使用 parts，删除原 sample 变量绑定；不删除缓存或 parts 中的数据。
    del sample

    # ------------------------------------------------------------------------
    # 03. 保存划分记录，供 Notebook 01 展示，并支持后续追踪数据来源。
    # *_row_ids.csv 保存原始行编号；splits.csv 有 3 个集合 × 2 个处理组共 6 行。
    # ------------------------------------------------------------------------
    split_summary = []

    for name, frame in parts.items():
        # 只选 row_id 数据列；index=False 表示不额外保存 DataFrame 的行索引。
        frame[['row_id']].to_csv(
            out / f'{name}_row_ids.csv',
            index=False,
        )

        # 在当前集合内分别统计 T=0、T=1；二元标签求和就是正例数。
        for a in [0, 1]:
            rows = frame[frame.treatment == a]
            split_summary.append(
                {
                    'split': name,
                    'treatment': a,
                    'n': len(rows),
                    'conversion_positive': int(rows.conversion.sum()),
                    'visit_positive': int(rows.visit.sum()),
                }
            )

    pd.DataFrame(
        split_summary
    ).to_csv(
        out / 'splits.csv',
        index=False,
    )

    # ------------------------------------------------------------------------
    # 04. 取出三个集合，并以训练集 treatment 均值估计总体分配概率 e。
    # e 供后续增量评价加权使用，不是实际曝光率，也不是结果转化率。
    # 它是常数的插入估计，并非外部给定的实验分流参数或按 X 拟合的倾向得分。
    # ------------------------------------------------------------------------
    train, val, test = (
        parts['train'],
        parts['validation'],
        parts['test'],
    )
    e = float(
        train.treatment.mean()
    )

    # ------------------------------------------------------------------------
    # 05. 特征重叠诊断：验证 / 测试集中有多少行的完整特征向量在训练集出现过？
    # 只使用 12 个 FEATURES，不包含 row_id、treatment 或结果标签。
    # 指纹基于建模用 float32 特征；64-bit hash 存在碰撞可能。
    # 相同匿名特征不等于相同用户，因此这里只记录风险线索，不据此删行。
    # ------------------------------------------------------------------------

    train_hash = pd.util.hash_pandas_object(
        train[FEATURES],
        index=False,
    ).to_numpy() # 训练集每行特征生成指纹

    # np.isin 逐行返回 True/False；mean() 得到匹配行比例，分母是当前集合行数。
    # 这不是“不同特征向量种类的重叠比例”，也不是 row_id 的重叠比例。
    overlap = {
        name: float(
            np.isin(
                pd.util.hash_pandas_object(
                    f[FEATURES],
                    index=False,
                ).to_numpy(),
                train_hash,
            ).mean()
        )
        for name, f in [
            ('validation', val),
            ('test', test),
        ]
    }
    write_json(
        {
            'note': 'Identical anonymized feature vectors, not identified users; based on modeling float32 representation',
            'fraction_matching_training_features': overlap,
        },
        out / 'feature_overlap.json',
    )

    # ------------------------------------------------------------------------
    # 06. 保存全量质量审计、全量 ATE 与样本比例失衡（SRM）诊断记录。
    # 这些汇总使用 audit 的全量计数，不是 train / validation / test 的局部计数。
    # Notebook 01、02 会读取相应文件；这里才是生成文件的位置。
    # ------------------------------------------------------------------------
    write_json(
        audit,
        out / 'audit.json',
    )
    pd.DataFrame(
        [
            aggregate_ate(audit, y)
            for y in ['conversion', 'visit']
        ]
    ).to_csv(
        out / 'ate_full.csv',
        index=False,
    )

    # ------------------------------------------------------------------------
    # 07. 实验可信度：全量公开数据 SRM、全量原始特征 SMD/KS，
    #     以及从原始全量来源组内等概率、两组等量抽样的 C2ST。
    # 与正式 Train/Validation/Test 无关；只更新原有五个诊断结果文件。
    # ------------------------------------------------------------------------
    update_experiment_diagnostics(
        args.data,
        out,
        audit=audit,
        source=source,
        sample_size=args.c2st_sample,
        permutations=args.permutations,
        seed=args.seed,
        progress=log,
        scanned=scanned,
    )
    del scanned

    # ------------------------------------------------------------------------
    # 08. 结果可学习性：在训练集拟合结果预测模型，在验证集评价 visit / conversion。
    # 判断 X 对结果 Y 有无预测信息；这不等于 X 能预测处理效应 CATE。
    # ------------------------------------------------------------------------
    learning = []
    for outcome in ['visit', 'conversion']:
        learning.append(
            learnability(
                train,
                val,
                FEATURES,
                outcome,
                seed=args.seed,
            )
        )
    pd.concat(
        learning
    ).to_csv(
        out / 'learnability.csv',
        index=False,
    )

    # ------------------------------------------------------------------------
    # 09. 在训练集拟合四个 Uplift 候选模型，并取得验证集上的预测与排序指标。
    # models 保存拟合对象；val_scores 保存各模型对每条验证样本的增量分数。
    # 每个模型写入独立 .joblib 文件；此阶段不计算测试集模型指标。
    # ------------------------------------------------------------------------
    models = {}
    val_scores = {}
    val_rank = []
    model_metadata = {}
    names = ['s_logistic', 't_logistic', 't_xgboost', 'x_xgboost']
    for name in names:
        log('训练 ' + name)
        # 只有 XGBoost 候选模型接收树数量参数；**kwargs 将字典展开为关键字参数。
        kwargs = (
            {'n_estimators': args.trees}
            if 'xgboost' in name
            else {}
        )
        model = make_model(
            name,
            random_state=args.seed,
            **kwargs,
        )
        started = time.perf_counter()
        model.fit(
            train[FEATURES],
            train.treatment,
            train[args.outcome],
        )
        model_metadata[name] = {
            'parameters': model.get_params(),
            'seconds': time.perf_counter() - started,
        }
        val_scores[name] = model.predict(
            val[FEATURES]
        )
        models[name] = model
        val_rank.append(
            {
                'policy': name,
                **ranking_metrics(
                    val[args.outcome],
                    val.treatment,
                    val_scores[name],
                    e,
                    args.seed,
                ),
            }
        )
        joblib.dump(
            model,
            out / f'{name}.joblib',
        )

    # ------------------------------------------------------------------------
    # 10. 在主预算下比较验证集策略，并冻结选择结果。
    # propensity 是响应倾向基线 P(Y=1 | X,T=1)，不是分配概率 P(T=1 | X)。
    # evaluate_policies 另行加入 random 基线；预算表示触达比例，不是货币金额。
    # ------------------------------------------------------------------------
    val_scores['propensity'] = models['t_xgboost'].predict_outcomes(
        val[FEATURES]
    )[1]
    val_policies = evaluate_policies(
        val[args.outcome],
        val.treatment,
        val_scores,
        e,
        budgets=[args.primary_budget],
        bootstrap=0,  # 验证集此处只用于方案比较，不计算 bootstrap 区间。
        seed=args.seed,
    )
    val_policies.to_csv(
        out / 'validation_policies.csv',
        index=False,
    )
    pd.DataFrame(
        val_rank
    ).to_csv(
        out / 'validation_ranking.csv',
        index=False,
    )

    # 按“每十万名可触达人群的增量结果数”降序，选出验证集表现最好的策略。
    # selected 可为基线或 Uplift 模型；selected_uplift 只在四个 Uplift 模型中选。
    # 所有选择在测试集评价之前确定；不利用测试集调参，也不合并 train+val 重拟合。
    selected = (
        val_policies.sort_values(
            'incremental_per_100k',
            ascending=False,
            kind='stable',
        )
        .iloc[0]
        .policy
    )
    selected_uplift = (
        val_policies[val_policies.policy.isin(names)]
        .sort_values(
            'incremental_per_100k',
            ascending=False,
            kind='stable',
        )
        .iloc[0]
        .policy
    )
    selection = {
        'primary_budget': args.primary_budget,
        'criterion': 'validation HT incremental responses per 100k eligible people',
        'selected_policy': selected,
        'selected_uplift_model': selected_uplift,
        'candidates': ['random', 'propensity', *names],
        'frozen_before_test': True,
        'note': 'All models remain trained on training split only; no test tuning, no train+validation refit.',
    }
    write_json(
        selection,
        out / 'selection.json',
    )

    # ------------------------------------------------------------------------
    # 11. 使用已拟合模型计算测试集分数、排序指标、绘图曲线和分组校准结果。
    # 所有候选方案均保留测试结果，但不据此重新选择最终策略。
    # ------------------------------------------------------------------------
    log(f'验证集选择已冻结：{selected}；开始一次独立测试集评估')
    test_scores = {
        name: model.predict(test[FEATURES])
        for name, model in models.items()
    }
    test_scores['propensity'] = models['t_xgboost'].predict_outcomes(
        test[FEATURES]
    )[1]
    rankings = []
    curves = []
    for name, score in test_scores.items():
        rankings.append(
            {
                'policy': name,
                **ranking_metrics(
                    test[args.outcome],
                    test.treatment,
                    score,
                    e,
                    args.seed,
                ),
                'mean_score': float(np.mean(score)),
                'negative_fraction': float(np.mean(score < 0)),
            }
        )
        curve = uplift_curve(
            test[args.outcome],
            test.treatment,
            score,
            e,
            args.seed,
        )

        # 排序指标使用完整曲线计算；这里只对另存的绘图表抽取最多 501 个位置。
        # 不对原始样本再抽样，也不会用这份简化表重新计算 AUUC / Qini。
        curve = curve.iloc[
            np.unique(
                np.linspace(
                    0,
                    len(curve) - 1,
                    501,
                ).astype(int)
            )
        ].copy()
        curve.insert(0, 'policy', name)
        curves.append(curve)
        # 只对增量分数做处理效应校准；响应概率分数不是同一种估计对象。
        if name != 'propensity':
            calibration_table(
                test[args.outcome],
                test.treatment,
                score,
            ).to_csv(
                out / f'calibration_{name}.csv',
                index=False,
            )
    pd.DataFrame(
        rankings
    ).to_csv(
        out / 'test_ranking.csv',
        index=False,
    )
    pd.concat(
        curves
    ).to_csv(
        out / 'curves.csv',
        index=False,
    )

    # ------------------------------------------------------------------------
    # 12. 预算约束下的测试集策略评价，以及 AUUC / Qini 的配对 bootstrap 区间。
    # 主预算已由验证集阶段固定；其余预算用于探索，不用于事后重新挑选主方案。
    # 区间以当前拟合模型、预测分数和 e 为条件，不包含训练和选模的不确定性。
    # ------------------------------------------------------------------------
    budgets = sorted(
        set(
            [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, args.primary_budget]
        )
    )
    policies = evaluate_policies(
        test[args.outcome],
        test.treatment,
        test_scores,
        e,
        budgets=budgets,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    policies.to_csv(
        out / 'test_policies.csv',
        index=False,
    )
    log('计算冻结模型的配对 AUUC / Qini 区间')
    ranking_confidence_intervals(
        test[args.outcome],
        test.treatment,
        test_scores,
        e,
        bootstrap=args.bootstrap,
        seed=args.seed,
    ).to_csv(
        out / 'ranking_intervals.csv',
        index=False,
    )

    # ------------------------------------------------------------------------
    # 13. 保存测试集平均效果与逐行预测，供后续评价、核对和补充分析复用。
    # ate_test 仅针对测试集的主结果变量，区别于全量两个结果变量的 ate_full。
    # predictions.npz 同时保留 row_id、真实 Y、真实 T 和每个模型 / 基线的分数。
    # ------------------------------------------------------------------------
    write_json(
        ate_summary(
            test[args.outcome],
            test.treatment,
        ),
        out / 'ate_test.json',
    )
    np.savez_compressed(
        out / 'predictions.npz',
        row_id=test.row_id.to_numpy(),
        y=test[args.outcome].to_numpy(),
        t=test.treatment.to_numpy(),
        **test_scores,
    )

    # ------------------------------------------------------------------------
    # 14. 保存本次实验说明 manifest：数据来源、参数、假设、模型选择和环境版本。
    # Notebook 的来源说明从这里读取；记录本身不等于已独立复现或验证因果假设。
    # 先记为 metrics_complete，报告生成成功后才更新为 complete。
    # ------------------------------------------------------------------------
    write_json(
        {
            'assignment_probability': e,
            'probability_source': 'training treatment fraction; plug-in estimate',
            'ci_method': 'paired within-arm percentile bootstrap, fixed learned scores and assignment probability; top-k reselected',
            'uncertainty_excluded': [
                'model training and selection',
                'assignment probability estimation',
                'unidentified user dependence',
            ],
            'estimand': 'public-sample assignment (ITT-style) incremental outcome relative to treat-none',
            'cost_assumption': 'equal cost/value; budget is contact fraction, not currency',
            'ranking_metric_version': 'v2; Qini retains control-only zero prefixes; plug-in AUUC explicit; no perfect normalization',
            'primary_outcome': args.outcome,
            'selection': selection,
            'models': model_metadata,
            'assumptions': [
                'released sample has approximately constant assignment probability conditional on X',
                'independent rows, consistency and no interference',
                'equal value per conversion and equal contact cost',
            ],
            'status': 'metrics_complete',
            'source': source,
            'features': FEATURES,
            'seed': args.seed,
            'sample_n': sum(
                len(v)
                for v in parts.values()
            ),
            'arguments': vars(args),
            'python': sys.version,
            'platform': platform.platform(),
            'versions': {
                name: importlib.metadata.version(name)
                for name in [
                    'numpy',
                    'pandas',
                    'scipy',
                    'scikit-learn',
                    'xgboost',
                    'matplotlib',
                    'joblib',
                ]
            },
            'elapsed_seconds': time.perf_counter() - start,
        },
        out / 'manifest.json',
    )

    # ------------------------------------------------------------------------
    # 15. 根据已保存的结果生成 HTML 报告和图片；这一步不重新训练模型。
    # 最后打印主预算下的策略表，并将本次输出目录返回给调用者。
    # ------------------------------------------------------------------------
    from criteo_uplift.reporting import build_report

    build_report(out)
    completion = json.loads(
        (out / 'manifest.json').read_text()
    )
    completion['status'] = 'complete'
    write_json(
        completion,
        out / 'manifest.json',
    )
    log(f'完成：{out}/report.html')
    print(
        policies[
            np.isclose(
                policies.budget,
                args.primary_budget,
            )
        ].to_string(
            index=False
        )
    )
    return out


# ============================================================================
# 命令行参数：决定分析对象、样本规模、随机种子、计算规模与结果目录。
# 不传某个选项时使用 default；这里解析配置，不执行分析。
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    # 数据与分析对象：原始 CSV 路径、抽样行数、主结果变量。
    # sample-size=0 表示保留全量用于建模，并非抽取零行；需额外考虑内存。
    p.add_argument(
        '--data',
        type=Path,
        default=DEFAULT_DATA,
    )
    p.add_argument(
        '--sample-size',
        type=int,
        default=1_000_000,
        help='0=全量',
    )
    p.add_argument(
        '--outcome',
        choices=['conversion', 'visit'],
        default='conversion',
    )

    # 随机种子传给抽样、划分和模型等步骤；主预算默认最多触达 20% 人群。
    p.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    p.add_argument(
        '--primary-budget',
        type=float,
        default=0.2,
    )

    # 诊断规模：全量来源的 C2ST 两组等量总样本数与完整重拟合置换次数。
    p.add_argument(
        '--permutations',
        type=int,
        default=499,
    )
    p.add_argument(
        '--c2st-sample',
        type=int,
        default=20000,
    )

    # 评价与模型规模：测试评价的 bootstrap 次数、XGBoost 的树数量参数。
    p.add_argument(
        '--bootstrap',
        type=int,
        default=200,
    )
    p.add_argument(
        '--trees',
        type=int,
        default=150,
    )

    # 可选的输出目录；未传入时由 run() 按实验配置构造默认路径。
    p.add_argument(
        '--output',
        type=Path,
    )

    p.add_argument(
        '--diagnostics-only',
        action='store_true',
        help='仅原位更新已有实验目录的全量诊断和诊断图/报告，不重新划分或训练模型',
    )

    # 解析终端输入为 Namespace；例如 --sample-size 对应 args.sample_size。
    args = p.parse_args()

    # 检查部分核心参数范围；不合法则报错退出，而不是继续运行耗时任务。
    if (
        args.sample_size < 0
        or not 0 < args.primary_budget <= 1
        or args.permutations < 1
        or args.c2st_sample < 20
        or args.c2st_sample % 20
        or args.bootstrap < 2
        or args.trees < 1
    ):
        p.error(
            'sample-size>=0, primary-budget in (0,1], permutations>=1, bootstrap>=2, trees>=1 required'
        )
    return args


# ============================================================================
# 脚本入口：直接执行本文件时，先读取参数，再启动 run()。
# 通过 import 导入本模块时不会进入这里，但顶部的运行环境配置仍会执行。
# ============================================================================
if __name__ == '__main__':
    args = parse_args()
    if args.diagnostics_only:
        out = safe_output(
            args.output
            or ROOT / f'artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}'
        )
        audit = json.loads((out / 'audit.json').read_text())
        manifest = json.loads((out / 'manifest.json').read_text())
        if args.seed != manifest['seed']:
            raise ValueError('诊断 seed 必须与当前实验一致')
        update_experiment_diagnostics(
            args.data, out, audit=audit, source=manifest['source'],
            sample_size=args.c2st_sample, permutations=args.permutations,
            seed=args.seed, progress=log,
        )
        # 同步现有 manifest 中的诊断参数；正式模型配置与选择记录保持原值。
        manifest['arguments']['permutations'] = args.permutations
        manifest['arguments']['c2st_sample'] = args.c2st_sample
        write_json(manifest, out / 'manifest.json')
        from criteo_uplift.reporting import build_report
        build_report(out, diagnostics_only=True)
        log(f'已原位更新诊断：{out}')
    else:
        run(args)
