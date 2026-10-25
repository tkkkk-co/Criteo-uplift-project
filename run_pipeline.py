#!/usr/bin/env python3
# 运行完整 Criteo uplift 分析：数据准备、诊断、训练、验证集选择、测试集评价和报告。
# 阅读地图：docs/PROJECT_PIPELINE.md；Notebook 与结果文件映射：docs/RESULTS_GUIDE.md。
# 阅读本文件时先看 run() 中的编号阶段，再沿当前阶段进入对应模块，不必先通读整个包。
# 原始 data 目录只读；运行示例：python run_pipeline.py --sample-size 1000000 --outcome conversion。

# 在导入 NumPy、pandas 或模型前配置数值计算线程。
from criteo_uplift.runtime import configure_runtime

configure_runtime()

import argparse
import json
import time
import numpy as np
from pathlib import Path

from criteo_uplift.data import ROOT, DEFAULT_DATA, split_sample
from criteo_uplift.diagnostics import update_experiment_diagnostics
from criteo_uplift.evaluation import aggregate_ate_from_audit as aggregate_ate
from criteo_uplift.preparation import prepare_sample as _prepare_sample
from criteo_uplift.training import (
    fit_and_select_models,
    run_learnability_diagnostics,
)
from criteo_uplift.holdout_evaluation import evaluate_test_set
from criteo_uplift.reporting import build_report
from criteo_uplift.pipeline_io import (
    safe_output,
    write_json,
    write_audit_artifacts,
    write_feature_overlap,
    write_run_manifest,
    write_split_artifacts,
)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：log —— 显示带时间的进度信息。
# 输入：
#     message：需要显示的进度文字。
# 输出：
#     返回值：None；向终端输出一条进度信息。
# 说明：显示带时间的进度信息。
#-----------------------------------------------------------------------------------------------------------------------
def log(message):
    import time as _time

    print(_time.strftime("%H:%M:%S"), message, flush=True)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：prepare_sample —— 核验缓存并准备建模样本和全量诊断。
# 输入：
#     args：本次运行的配置对象，包含数据路径、规模、种子和输出设置。
# 输出：
#     sample：建模样本表。
#     audit：全量审计字典。
#     source：原始文件指纹。
#     scanned：同一次扫描得到的可复用诊断结果。
# 说明：保留原有导入入口，将实现转交 preparation.prepare_sample；显式传入项目根目录和进度函数。
#-----------------------------------------------------------------------------------------------------------------------
def prepare_sample(args):
    return _prepare_sample(args, root=ROOT, progress=log)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：finalize_run —— 生成报告并记录本次实验已完成。
# 输入：
#     out：当前实验的结果保存目录。
#     policies：测试集各策略的评价结果表。
#     primary_budget：主策略允许触达的人群比例。
#     progress：接收进度文字的函数。
# 输出：
#     out：已生成报告的实验目录。
# 说明：根据已保存的产物生成报告，更新完成状态，并显示主预算下的策略结果。
#-----------------------------------------------------------------------------------------------------------------------
def finalize_run(out, policies, primary_budget, progress=print):
    out = Path(out)

    build_report(out)
    manifest_path = out / "manifest.json"
    completion = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion["status"] = "complete"
    write_json(completion, manifest_path)
    progress(f"完成：{out}/report.html")
    print(
        policies[
            np.isclose(policies.budget, primary_budget)
        ].to_string(index=False)
    )
    return out


#-----------------------------------------------------------------------------------------------------------------------
# 函数：update_diagnostics_only —— 更新已有实验的诊断文件和报告。
# 输入：
#     args：本次运行的配置对象，包含数据路径、规模、种子和输出设置。
#     root：项目根目录，用于定位缓存或结果。
#     progress：接收进度文字的函数。
# 输出：
#     out：完成诊断更新的已有实验目录。
# 说明：复用实验审计与来源信息，仅重做诊断，不划分或训练正式增量模型。
#-----------------------------------------------------------------------------------------------------------------------
def update_diagnostics_only(args, *, root=ROOT, progress=print):
    out = safe_output(
        args.output
        or Path(root) / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.seed != manifest["seed"]:
        raise ValueError("诊断 seed 必须与当前实验一致")
    update_experiment_diagnostics(
        args.data,
        out,
        audit=audit,
        source=manifest["source"],
        sample_size=args.c2st_sample,
        permutations=args.permutations,
        seed=args.seed,
        progress=progress,
    )
    manifest["arguments"]["permutations"] = args.permutations
    manifest["arguments"]["c2st_sample"] = args.c2st_sample
    write_json(manifest, manifest_path)
    build_report(out, diagnostics_only=True)
    progress(f"已原位更新诊断：{out}")
    return out


#-----------------------------------------------------------------------------------------------------------------------
# 函数：run —— 串联完整实验流程。
# 输入：
#     args：本次运行的配置对象，包含数据路径、规模、种子和输出设置。
# 输出：
#     out：报告成功生成后的实验结果目录。
# 说明：依次准备数据、检查分组、拟合模型、冻结验证集选择、评价测试集并生成报告。
#-----------------------------------------------------------------------------------------------------------------------
def run(args):
    started_at = time.perf_counter()

    # 01. 为本次实验创建新的结果目录。
    out = safe_output(
        args.output
        or ROOT / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"结果目录已有正式实验：{out}。仅更新诊断请使用 --diagnostics-only。"
        )
    out.mkdir(parents=True, exist_ok=True)

    # 02. 准备建模样本、全量审计、来源指纹和同次扫描的诊断结果。
    sample, audit, source, scanned = prepare_sample(args)
    parts = split_sample(sample, args.outcome, args.seed)
    del sample

    # 03. 保存划分行号与汇总，并检查不同集合间相同特征向量的重叠。
    write_split_artifacts(parts, out)
    train, validation, test = (
        parts["train"],
        parts["validation"],
        parts["test"],
    )
    assignment_probability = float(train.treatment.mean())
    write_feature_overlap(parts, out)

    # 04. 保存全量审计及汇总 ATE 结果。
    write_audit_artifacts(audit, out)

    # 05. 执行 SRM、全量 SMD/KS 和 C2ST；复用扫描结果避免重复遍历。
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

    # 06. 检查处理前特征能否预测观察到的结果。
    run_learnability_diagnostics(
        train, validation, out, seed=args.seed
    )

    # 07. 在训练集拟合候选模型，仅用验证集冻结策略选择。
    fitted = fit_and_select_models(
        train,
        validation,
        outcome=args.outcome,
        seed=args.seed,
        trees=args.trees,
        primary_budget=args.primary_budget,
        assignment_probability=assignment_probability,
        out=out,
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
        out=out,
        progress=log,
    )

    # 09. 保存可复现信息、生成报告，并标记实验完成。
    write_run_manifest(
        out,
        args=args,
        assignment_probability=assignment_probability,
        selection=fitted["selection"],
        model_metadata=fitted["model_metadata"],
        source=source,
        sample_n=sum(len(frame) for frame in parts.values()),
        started_at=started_at,
    )
    return finalize_run(out, policies, args.primary_budget, progress=log)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：parse_args —— 解析并检查命令行配置。
# 输入：
#     argv：命令行参数列表；None 表示读取当前进程的参数。
# 输出：
#     args：解析后的命令行配置对象。
# 说明：解析并检查命令行配置。
#-----------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='运行完整 Criteo uplift 分析：数据准备、诊断、训练、验证集选择、测试集评价和报告。\n原始 data 目录只读；运行示例：python run_pipeline.py --sample-size 1000000 --outcome conversion。'
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
    args = p.parse_args(argv)

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


#-----------------------------------------------------------------------------------------------------------------------
# 函数：main —— 根据命令行配置启动任务。
# 输入：
#     argv：命令行参数列表；None 表示读取当前进程的参数。
# 输出：
#     返回值：None；执行请求的流程并保存或显示结果。
# 说明：根据命令行配置启动任务。
#-----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    if args.diagnostics_only:
        update_diagnostics_only(args, root=ROOT, progress=log)
    else:
        run(args)


if __name__ == "__main__":
    main()
