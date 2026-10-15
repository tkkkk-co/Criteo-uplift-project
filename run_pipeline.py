#!/usr/bin/env python3
"""运行：python run_pipeline.py --sample-size 1000000 --outcome conversion

输出仅 artifacts/，data/ 永远只读。--sample-size 0 为全量建模（较耗内存）。
默认复用校验过来源 SHA256 的随机样本；模型与结果每次写入独立 run 目录。
"""

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
    scan_and_sample,
    save_sample,
    load_sample,
    split_sample,
    safe_output,
    write_json,
    fingerprint,
)
from criteo_uplift.models import make_model
from criteo_uplift.diagnostics import covariate_balance, c2st, learnability, srm_summary
from criteo_uplift.evaluation import (
    ate_summary,
    ranking_metrics,
    uplift_curve,
    evaluate_policies,
    calibration_table,
    ranking_confidence_intervals,
)


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def aggregate_ate(audit, outcome):
    """全量审计只保存聚合量，按二项数据精确重建均值方差。"""
    from scipy.stats import norm

    n0, n1 = (audit['groups'][str(a)]['n'] for a in [0, 1])
    r0, r1 = (audit['groups'][str(a)][outcome]['positive'] for a in [0, 1])
    p0, p1 = r0 / n0, r1 / n1
    delta = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / (n1 - 1) + p0 * (1 - p0) / (n0 - 1))
    pooled = (r0 + r1) / (n0 + n1)
    nullse = np.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
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
        'p_value': 2 * norm.sf(abs(delta / nullse)) if nullse > 0 else 1.0,
        'relative_uplift': delta / p0 if p0 else None,
    }


def prepare_sample(args):
    base = safe_output(ROOT / 'artifacts')
    base.mkdir(exist_ok=True)
    cache = base / f'cache/sample_{args.sample_size}_seed{args.seed}.npz'
    metadata = cache.with_suffix('.json')
    current = fingerprint(args.data)
    reuse = False
    if cache.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        reuse = (
            info['source']['sha256'] == current['sha256']
            and info['sample_size'] == args.sample_size
            and info['seed'] == args.seed
        )
    # Earlier explicit scan has identical provenance, so adopt that cache once.
    initial = base / 'source_fingerprint.json'
    if (
        not metadata.exists()
        and cache.exists()
        and initial.exists()
        and args.sample_size == 1_000_000
        and args.seed == 42
    ):
        reuse = json.loads(initial.read_text())['sha256'] == current['sha256']
    if reuse:
        log('复用已校验来源 SHA256 的随机样本')
        sample = load_sample(cache)
        audit = json.loads(
            (metadata if metadata.exists() else base / 'full_audit.json').read_text()
        )
        if 'audit' in audit:
            audit = audit['audit']
    else:
        log('全量扫描 + 等概率随机抽样')
        sample, audit = scan_and_sample(args.data, args.sample_size, args.seed)
        save_sample(sample, cache)
    write_json(
        {
            'source': current,
            'sample_size': args.sample_size,
            'seed': args.seed,
            'audit': audit,
        },
        metadata,
    )
    return sample, audit, current


def run(args):
    start = time.perf_counter()
    out = safe_output(
        args.output
        or ROOT / f'artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}'
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f'结果目录非空，防止混合不同实验：{out}。请用 --output 指定新目录。'
        )
    out.mkdir(parents=True, exist_ok=True)
    sample, audit, source = prepare_sample(args)
    parts = split_sample(sample, args.outcome, args.seed)
    del sample
    split_summary = []
    for name, frame in parts.items():
        frame[['row_id']].to_csv(out / f'{name}_row_ids.csv', index=False)
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
    pd.DataFrame(split_summary).to_csv(out / 'splits.csv', index=False)
    train, val, test = parts['train'], parts['validation'], parts['test']
    e = float(train.treatment.mean())
    # Matching anonymous features is not evidence of the same person.
    train_hash = pd.util.hash_pandas_object(train[FEATURES], index=False).to_numpy()
    overlap = {
        name: float(
            np.isin(
                pd.util.hash_pandas_object(f[FEATURES], index=False).to_numpy(),
                train_hash,
            ).mean()
        )
        for name, f in [('validation', val), ('test', test)]
    }
    write_json(
        {
            'note': 'Identical anonymized feature vectors, not identified users; based on modeling float32 representation',
            'fraction_matching_training_features': overlap,
        },
        out / 'feature_overlap.json',
    )
    write_json(audit, out / 'audit.json')
    pd.DataFrame([aggregate_ate(audit, y) for y in ['conversion', 'visit']]).to_csv(
        out / 'ate_full.csv', index=False
    )
    write_json(srm_summary(audit['n_treatment'], audit['n_control']), out / 'srm.json')
    log('随机化与可学习性诊断（仅使用 train/validation）')
    covariate_balance(train, FEATURES).to_csv(out / 'balance.csv', index=False)
    check = c2st(
        train,
        FEATURES,
        sample_size=args.c2st_sample,
        permutations=args.permutations,
        seed=args.seed,
    )
    check['summary'].to_csv(out / 'c2st.csv', index=False)
    check['null_auc'].to_csv(out / 'c2st_null.csv', index=False)
    write_json(
        {k: v for k, v in check.items() if not isinstance(v, pd.DataFrame)},
        out / 'c2st_details.json',
    )
    learning = []
    for outcome in ['visit', 'conversion']:
        learning.append(learnability(train, val, FEATURES, outcome, seed=args.seed))
    pd.concat(learning).to_csv(out / 'learnability.csv', index=False)
    models = {}
    val_scores = {}
    val_rank = []
    model_metadata = {}
    names = ['s_logistic', 't_logistic', 't_xgboost', 'x_xgboost']
    for name in names:
        log('训练 ' + name)
        kwargs = {'n_estimators': args.trees} if 'xgboost' in name else {}
        model = make_model(name, random_state=args.seed, **kwargs)
        started = time.perf_counter()
        model.fit(train[FEATURES], train.treatment, train[args.outcome])
        model_metadata[name] = {
            'parameters': model.get_params(),
            'seconds': time.perf_counter() - started,
        }
        val_scores[name] = model.predict(val[FEATURES])
        models[name] = model
        val_rank.append(
            {
                'policy': name,
                **ranking_metrics(
                    val[args.outcome], val.treatment, val_scores[name], e, args.seed
                ),
            }
        )
        joblib.dump(model, out / f'{name}.joblib')
    # Response propensity = P(Y=1 | X,T=1); it is not P(T=1 | X).
    val_scores['propensity'] = models['t_xgboost'].predict_outcomes(val[FEATURES])[1]
    val_policies = evaluate_policies(
        val[args.outcome],
        val.treatment,
        val_scores,
        e,
        budgets=[args.primary_budget],
        bootstrap=0,
        seed=args.seed,
    )
    val_policies.to_csv(out / 'validation_policies.csv', index=False)
    pd.DataFrame(val_rank).to_csv(out / 'validation_ranking.csv', index=False)
    # Fix all choices before computing any test model metric.
    selected = (
        val_policies.sort_values('incremental_per_100k', ascending=False, kind='stable')
        .iloc[0]
        .policy
    )
    selected_uplift = (
        val_policies[val_policies.policy.isin(names)]
        .sort_values('incremental_per_100k', ascending=False, kind='stable')
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
    write_json(selection, out / 'selection.json')
    log(f'验证集选择已冻结：{selected}；开始一次独立测试集评估')
    test_scores = {
        name: model.predict(test[FEATURES]) for name, model in models.items()
    }
    test_scores['propensity'] = models['t_xgboost'].predict_outcomes(test[FEATURES])[1]
    rankings = []
    curves = []
    for name, score in test_scores.items():
        rankings.append(
            {
                'policy': name,
                **ranking_metrics(
                    test[args.outcome], test.treatment, score, e, args.seed
                ),
                'mean_score': float(np.mean(score)),
                'negative_fraction': float(np.mean(score < 0)),
            }
        )
        curve = uplift_curve(test[args.outcome], test.treatment, score, e, args.seed)
        # Full resolution is used for integration; plotting table is downsampled.
        curve = curve.iloc[
            np.unique(np.linspace(0, len(curve) - 1, 501).astype(int))
        ].copy()
        curve.insert(0, 'policy', name)
        curves.append(curve)
        if name != 'propensity':
            calibration_table(test[args.outcome], test.treatment, score).to_csv(
                out / f'calibration_{name}.csv', index=False
            )
    pd.DataFrame(rankings).to_csv(out / 'test_ranking.csv', index=False)
    pd.concat(curves).to_csv(out / 'curves.csv', index=False)
    budgets = sorted(set([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, args.primary_budget]))
    policies = evaluate_policies(
        test[args.outcome],
        test.treatment,
        test_scores,
        e,
        budgets=budgets,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    policies.to_csv(out / 'test_policies.csv', index=False)
    log('计算冻结模型的配对 AUUC / Qini 区间')
    ranking_confidence_intervals(
        test[args.outcome],
        test.treatment,
        test_scores,
        e,
        bootstrap=args.bootstrap,
        seed=args.seed,
    ).to_csv(out / 'ranking_intervals.csv', index=False)
    write_json(ate_summary(test[args.outcome], test.treatment), out / 'ate_test.json')
    np.savez_compressed(
        out / 'predictions.npz',
        row_id=test.row_id.to_numpy(),
        y=test[args.outcome].to_numpy(),
        t=test.treatment.to_numpy(),
        **test_scores,
    )
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
            'sample_n': sum(len(v) for v in parts.values()),
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
    from criteo_uplift.reporting import build_report

    build_report(out)
    completion = json.loads((out / 'manifest.json').read_text())
    completion['status'] = 'complete'
    write_json(completion, out / 'manifest.json')
    log(f'完成：{out}/report.html')
    print(
        policies[np.isclose(policies.budget, args.primary_budget)].to_string(
            index=False
        )
    )
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--sample-size', type=int, default=1_000_000, help='0=全量')
    p.add_argument('--outcome', choices=['conversion', 'visit'], default='conversion')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--primary-budget', type=float, default=0.2)
    p.add_argument('--permutations', type=int, default=99)
    p.add_argument('--c2st-sample', type=int, default=20000)
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--trees', type=int, default=150)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if (
        args.sample_size < 0
        or not 0 < args.primary_budget <= 1
        or args.permutations < 1
        or args.bootstrap < 2
        or args.trees < 1
    ):
        p.error(
            'sample-size>=0, primary-budget in (0,1], permutations>=1, bootstrap>=2, trees>=1 required'
        )
    return args


if __name__ == '__main__':
    run(parse_args())
