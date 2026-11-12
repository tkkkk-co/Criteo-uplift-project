#!/usr/bin/env python3
#-------------
# 项目主流程：准备 → 诊断 → 训练/冻结 → Test → 稳定性 → 业务规划 → 报告。
#-------------
from criteo_uplift.config import configure_runtime, log
configure_runtime()

import argparse
import json
from pathlib import Path
import time
import numpy as np
from criteo_uplift.config import ROOT, DEFAULT_DATA, DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, FEATURES
from criteo_uplift.preparation import prepare_sample, split_sample
from criteo_uplift.diagnostics import run_diagnostics
from criteo_uplift.training_validation import fit_and_select_models
from criteo_uplift.evaluation import ate_summary_from_audit, evaluate_test_set
from criteo_uplift.stability import run_stability, DEFAULT_SEEDS, DEFAULT_FRACTIONS
from criteo_uplift.planning import run_planning
from criteo_uplift.reporting import finalize_run, save_sample_feature_plot
from criteo_uplift.pipeline_io import safe_output, write_result


def run(args):
    #-------------
    # 函数名：run
    # 功能：按既定顺序执行数据准备、诊断、ATE、模型训练验证、Test 评价、Stability、Planning 和最终报告生成。
    # 输入参数：
    #     - args：parse_args 生成的完整命令行参数对象。
    # 输出参数：
    #     - 本次实验输出目录 pathlib.Path。
    #-------------
    started = time.perf_counter()

    # 01. 一次规划全部输出；缓存共享，正式结果按实验隔离。
    out = safe_output(args.output or ROOT / f'artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'结果目录非空：{out}。请用 --output 指定新的实验目录。')
    manifest_path = out / 'manifest.json'  # 来源、参数、冻结选择和完成状态
    report_path = out / 'report.html'  # 展示完整报告
    data_out = out / 'data_preparation'  # 全量审计、抽样说明、划分行号
    diagnostics_out = out / 'diagnostics'  # SRM、SMD/KS、C2ST、可学习性
    models_out = out / 'models'  # 模型、Validation 评价、冻结选择
    evaluation_out = out / 'evaluation'  # 统一 ATE、Test 预测、指标与区间
    stability_out = out / 'stability'  # 种子、训练规模和划分方式敏感性
    business_out = out / 'business'  # 假设参数、A/B 规划、经济情景
    figures_out = out / 'figures'  # 报告和 Notebook 使用的结果图
    sample_cache = ROOT / f'artifacts/cache/sample_{args.sample_size}_seed{args.seed}.npz'
    for directory in (data_out, diagnostics_out, models_out, evaluation_out, stability_out, business_out, figures_out):
        directory.mkdir(parents=True, exist_ok=True)
    assumptions = json.loads(args.planning_assumptions.read_text()) if args.planning_assumptions else None

    # 02. 一次流式扫描完成检查、审计与抽样，随后划分 60/20/20。
    sample, audit, source, scanned = prepare_sample(args, root=ROOT, progress=log)
    sample_n = len(sample)
    parts, summary, overlap = split_sample(sample, args.outcome, args.seed)
    train, validation, test = (parts[name] for name in ('train', 'validation', 'test'))
    for name, frame in parts.items():
        write_result(frame[['row_id']], data_out / f'{name}_row_ids.csv')
    write_result(summary, data_out / 'split_summary.csv')
    write_result(overlap, data_out / 'feature_overlap.json')
    write_result(audit, data_out / 'audit.json')
    write_result(scanned['sample_report'], data_out / 'sample_report.json')
    save_sample_feature_plot(sample, figures_out / 'sample_features.png')
    del sample

    # 03. 复用扫描统计与 C2ST 样本，完成分组诊断和可学习性检查。
    run_diagnostics(diagnostics_out, audit=audit, source=source, scanned=scanned,
                    train=train, validation=validation, sample_size=args.c2st_sample,
                    permutations=args.permutations, seed=args.seed, progress=log)
    del scanned

    # 04. 全量 ATE 只用审计计数，后面与 Test ATE 合并保存。
    full_ate = [{'scope': 'full_dataset', 'outcome': outcome, **ate_summary_from_audit(audit, outcome)}
                for outcome in ('conversion', 'visit')]

    # 05. Train 拟合四个候选，Validation 选型并冻结，Test 不参与。
    e = float(train.treatment.mean())
    fitted = fit_and_select_models(train, validation, outcome=args.outcome, seed=args.seed,
                                  trees=args.trees, primary_budget=args.primary_budget,
                                  assignment_probability=e, out=models_out, progress=log)
    # 06. 固定模型评价 Test；统一保存两种 scope、两种 outcome 的 ATE。
    policies = evaluate_test_set(test, fitted['models'], outcome=args.outcome,
                                 assignment_probability=e, seed=args.seed, bootstrap=args.bootstrap,
                                 primary_budget=args.primary_budget, out=evaluation_out,
                                 progress=log, full_ate=full_ate)
    del train, validation, test, parts

    # 07. 排除原 Test，按原设计检查随机种子、训练规模、分组划分。
    run_stability(sample_cache, out, stability_out, seeds=args.stability_seeds,
                  fractions=args.stability_fractions, trees=args.trees,
                  primary_budget=args.primary_budget, outcome=args.outcome)

    # 08. 使用明确的假设做线上 A/B 规划和条件性经济价值测算。
    run_planning(policies, fitted['selection'], business_out, assumptions=assumptions)
    # 09. 汇总已有结果，生成唯一正式 HTML，成功后标记实验完成。
    finalize_run(manifest_path=manifest_path, report_path=report_path, args=args, source=source,
                 sample_n=sample_n, assignment_probability=e, fitted=fitted,
                 elapsed_seconds=time.perf_counter() - started)
    log(f'完成：{report_path}')
    print(policies[np.isclose(policies.budget, args.primary_budget)].to_string(index=False))
    return out


def parse_args(argv=None):
    #-------------
    # 函数名：parse_args
    # 功能：解析 Pipeline 命令行参数，并检查抽样规模、预算、随机种子、Bootstrap 和 Stability 配置边界。
    # 输入参数：
    #     - argv：可选的命令行参数字符串列表；None 表示读取终端参数。
    # 输出参数：
    #     - 校验通过的 argparse.Namespace 参数对象。
    #-------------
    parser = argparse.ArgumentParser(description='完整 Criteo uplift 流程；原始 data 只读，输出拒绝覆盖。')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE, help='0=全量')
    parser.add_argument('--outcome', choices=['conversion', 'visit'], default='conversion')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--primary-budget', type=float, default=.2)
    parser.add_argument('--permutations', type=int, default=499)
    parser.add_argument('--c2st-sample', type=int, default=20000)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--trees', type=int, default=150)
    parser.add_argument('--stability-seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--stability-fractions', type=float, nargs='+', default=DEFAULT_FRACTIONS)
    parser.add_argument('--planning-assumptions', type=Path, help='完整假设 JSON；默认沿用 planning.py 中明确标记的原教学情景')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if (args.sample_size < 0 or not 0 < args.primary_budget <= 1 or args.permutations < 1
        or args.c2st_sample < 20 or args.c2st_sample % 20 or args.bootstrap < 2 or args.trees < 1
        or not 0 <= args.seed < 2**32 - 1):
        parser.error('sample-size>=0, budget in (0,1], permutations>=1, c2st-sample>=20 and divisible by 20, bootstrap>=2, trees>=1, valid seed required')
    from criteo_uplift.stability import _validate_seeds, _validate_fractions
    try:
        _validate_seeds(args.stability_seeds)
        _validate_fractions(args.stability_fractions)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    #-------------
    # 函数名：main
    # 功能：作为命令行入口解析参数并启动完整 Pipeline。
    # 输入参数：
    #     - argv：可选的命令行参数字符串列表；None 表示读取终端参数。
    # 输出参数：
    #     - run 返回的本次实验输出目录 pathlib.Path。
    #-------------
    return run(parse_args(argv))


if __name__ == '__main__':
    main()
