#!/usr/bin/env python3
"""完整实验入口。先读 run() 的编号阶段，再进入对应模块；见 docs/PROJECT_PIPELINE.md。"""

# 在导入数值计算库前配置线程。
from criteo_uplift.runtime import configure_runtime, log

configure_runtime()

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from criteo_uplift.preparation import DEFAULT_DATA, ROOT, prepare_sample, split_sample
from criteo_uplift.diagnostic_refresh import update_diagnostics_only
from criteo_uplift.diagnostics import update_experiment_diagnostics, run_learnability_diagnostics
from criteo_uplift.evaluation import aggregate_ate_from_audit
from criteo_uplift.training import fit_and_select_models
from criteo_uplift.holdout_evaluation import evaluate_test_set
from criteo_uplift.reporting import finalize_run, save_sample_feature_plot
from criteo_uplift.pipeline_io import safe_output, write_result


def run(args):
    """依次准备数据、诊断、训练、冻结策略、测试评价和生成报告。"""
    started_at = time.perf_counter()

    # 01. 规划本次实验的输出路径，创建总目录和各阶段子目录。
    out = safe_output(
        args.output
        or ROOT / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"结果目录已有正式实验：{out}。仅更新诊断请使用 --diagnostics-only。"
        )
    manifest_path = out / "manifest.json"  # 实验参数、数据来源与完成状态
    report_path = out / "report.html"  # 实验总报告
    data_out = out / "data_preparation"  # 全量审计、抽样与划分记录
    diagnostics_out = out / "diagnostics"  # 全量 ATE 与各项诊断结果
    models_out = out / "models"  # 训练好的模型与验证集选型结果
    evaluation_out = out / "evaluation"  # 测试集预测、评价指标与区间
    figures_out = out / "figures"  # 报告和 Notebook 展示用图
    out.mkdir(parents=True, exist_ok=True)
    for directory in (data_out, diagnostics_out, models_out, evaluation_out, figures_out):
        directory.mkdir(exist_ok=True)

    # 02. 统一扫描原始数据，准备审计、诊断所需数据和建模样本，再划分样本。
    sample, audit, source, scanned = prepare_sample(args, root=ROOT, progress=log)
    sample_n = len(sample)
    parts, split_summary, feature_overlap = split_sample(sample, args.outcome, args.seed)
    train, validation, test = (parts[name] for name in ("train", "validation", "test"))

    # 03. 保存数据准备记录和样本特征分布图。
    for name, frame in parts.items():
        write_result(frame[["row_id"]], data_out / f"{name}_row_ids.csv")
    write_result(split_summary, data_out / "split_summary.csv")
    write_result(feature_overlap, data_out / "feature_overlap.json")
    write_result(audit, data_out / "audit.json")
    write_result(scanned["sample_report"], data_out / "sample_report.json")
    save_sample_feature_plot(sample, figures_out / "sample_features.png")
    del sample

    # 04. 根据全量审计计数计算并保存 conversion 和 visit 的 ATE。
    ate_full = pd.DataFrame([
        aggregate_ate_from_audit(audit, outcome) for outcome in ("conversion", "visit")
    ])
    write_result(ate_full, diagnostics_out / "ate_full.csv")

    # 05. 运行 C2ST，和第 02 步算出的 SRM、SMD/KS 一起保存。
    update_experiment_diagnostics(
        args.data,
        diagnostics_out,
        audit=audit,
        source=source,
        sample_size=args.c2st_sample,
        permutations=args.permutations,
        seed=args.seed,
        progress=log,
        scanned=scanned,
    )
    del scanned

    # 06. 检查处理前特征能否预测观察到的结果。
    run_learnability_diagnostics(
        train, validation, diagnostics_out, seed=args.seed
    )

    # 07. 在训练集拟合候选模型，仅用验证集冻结策略选择。
    assignment_probability = float(train.treatment.mean())
    fitted = fit_and_select_models(
        train,
        validation,
        outcome=args.outcome,
        seed=args.seed,
        trees=args.trees,
        primary_budget=args.primary_budget,
        assignment_probability=assignment_probability,
        out=models_out,
        progress=log,
    )

    # 08. 在测试集评价冻结模型并保存测试结果。
    policies = evaluate_test_set(
        test,
        fitted["models"],
        outcome=args.outcome,
        assignment_probability=assignment_probability,
        seed=args.seed,
        bootstrap=args.bootstrap,
        primary_budget=args.primary_budget,
        out=evaluation_out,
        progress=log,
    )

    # 09. 写入 manifest.json，生成 report.html，再标记实验完成。
    finalize_run(
        manifest_path=manifest_path,
        report_path=report_path,
        args=args,
        source=source,
        sample_n=sample_n,
        assignment_probability=assignment_probability,
        fitted=fitted,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    log(f"完成：{report_path}")
    print(policies[np.isclose(policies.budget, args.primary_budget)].to_string(index=False))
    return out


def parse_args(argv=None):
    """解析并检查命令行配置。"""
    parser = argparse.ArgumentParser(description="运行完整 Criteo uplift 分析；原始 data 目录只读。")
    # 数据、抽样与分析对象。
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample-size", type=int, default=1_000_000, help="0=全量")
    parser.add_argument("--outcome", choices=["conversion", "visit"], default="conversion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-budget", type=float, default=0.2)
    # 诊断、模型与评价规模。
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--c2st-sample", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--trees", type=int, default=150)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--diagnostics-only", action="store_true",
        help="更新已有实验的审计、ATE 尾概率、诊断和报告；复用划分与正式模型",
    )
    args = parser.parse_args(argv)
    if (
        args.sample_size < 0
        or not 0 < args.primary_budget <= 1
        or args.permutations < 1
        or args.c2st_sample < 20
        or args.c2st_sample % 20
        or args.bootstrap < 2
        or args.trees < 1
    ):
        parser.error(
            "sample-size>=0, primary-budget in (0,1], permutations>=1, "
            "c2st-sample>=20 and divisible by 20, bootstrap>=2, trees>=1 required"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.diagnostics_only:
        update_diagnostics_only(args, root=ROOT, progress=log)
    else:
        run(args)


if __name__ == "__main__":
    main()
