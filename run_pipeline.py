#!/usr/bin/env python3
# 运行完整 Criteo uplift 分析：数据准备、诊断、训练、验证集选择、测试集评价和报告。
# 阅读地图：docs/PROJECT_PIPELINE.md；Notebook 与结果文件映射：docs/RESULTS_GUIDE.md。
# 阅读本文件时先看 run() 中的编号阶段，再沿当前阶段进入对应模块，不必先通读整个包。
# 原始 data 目录只读；运行示例：python run_pipeline.py --sample-size 1000000 --outcome conversion。

# 在导入 NumPy、pandas 或模型前配置数值计算线程。
from criteo_uplift.runtime import configure_runtime

configure_runtime()

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

from criteo_uplift.data import ROOT, DEFAULT_DATA, FEATURES, split_sample
from criteo_uplift.diagnostics import update_experiment_diagnostics
from criteo_uplift.evaluation import aggregate_ate_from_audit as aggregate_ate
from criteo_uplift.preparation import prepare_sample as _prepare_sample
from criteo_uplift.training import (
    fit_and_select_models,
    run_learnability_diagnostics,
)
from criteo_uplift.holdout_evaluation import evaluate_test_set
from criteo_uplift.reporting import build_report, save_sample_feature_plot
from criteo_uplift.pipeline_io import (
    safe_output,
    write_result,
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
    write_result(completion, manifest_path)
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
# 说明：沿现有入口补齐审计和数值稳定的 ATE 字段，重做诊断；使用已保存行号，不重新划分或训练正式增量模型。
#-----------------------------------------------------------------------------------------------------------------------
def update_diagnostics_only(args, *, root=ROOT, progress=print):
    out = safe_output(
        args.output
        or Path(root) / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    audit = json.loads(((out / 'data_preparation' / 'audit.json')).read_text(encoding="utf-8"))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.seed != manifest["seed"]:
        raise ValueError("诊断 seed 必须与当前实验一致")
    # 缓存缺少审计字段时，prepare_sample 按正常失效规则在同次扫描中重建。
    saved_sample_report = json.loads((out / "data_preparation/sample_report.json").read_text())
    prepare_args = argparse.Namespace(**{**vars(args), "sample_size": saved_sample_report["sample_size"]})
    sample, refreshed_audit, source, scanned = _prepare_sample(prepare_args, root=root, progress=progress)
    if source["sha256"] != manifest["source"]["sha256"]:
        raise ValueError("当前数据与已有实验的来源指纹不同")
    for key in ("n_rows", "n_treatment", "n_control", "duplicate_row_fingerprints"):
        if refreshed_audit[key] != audit[key]:
            raise ValueError(f"重新审计与已有实验计数不一致：{key}")
    audit = refreshed_audit
    write_result(audit, out / "data_preparation/audit.json")
    write_result(scanned["sample_report"], out / "data_preparation/sample_report.json")

    # 保留已有 ATE、标准误和区间，只补齐同一 pooled z 检验的稳定尾概率字段。
    ate_path = out / "diagnostics/ate_full.csv"
    ate_full = pd.read_csv(ate_path, dtype={"p_value_text": "string"}, float_precision="round_trip")
    for index, row in ate_full.iterrows():
        result = aggregate_ate(audit, row["outcome"])
        for key in ("n", "n_control", "n_treatment", "response_control", "response_treatment", "ate"):
            if not np.isclose(row[key], result[key], rtol=1e-12, atol=1e-15):
                raise ValueError(f"已有 ATE 与审计不一致：{row['outcome']} / {key}")
        for key in ("p_value", "log_p_value", "p_value_text"):
            ate_full.loc[index, key] = result[key]
    write_result(ate_full, ate_path)

    # 替换分类器时把已有 HGB 诊断保存在同一个详情文件中，避免丢失历史统计。
    old_details = json.loads((out / "diagnostics/c2st_details.json").read_text())
    old_c2st = pd.read_csv(out / "diagnostics/c2st.csv")
    previous_hgb = old_details.get("previous_hgb_diagnostics")
    if "hgb" in set(old_c2st["model"]):
        previous_hgb = {
            "note": "历史 HGB 配置结果；未按 XGBoost 重算，不参与当前两模型 Holm 校正。",
            "details": old_details,
            **{name: pd.read_csv(out / f"diagnostics/{name}.csv").to_dict("records")
               for name in ("c2st", "c2st_null", "learnability")},
        }
    update_experiment_diagnostics(
        args.data,
        out / "diagnostics",
        audit=audit,
        source=source,
        sample_size=args.c2st_sample,
        permutations=args.permutations,
        seed=args.seed,
        progress=progress,
        scanned=scanned,
    )
    if previous_hgb is not None:
        details_path = out / "diagnostics/c2st_details.json"
        details = json.loads(details_path.read_text())
        details["previous_hgb_diagnostics"] = previous_hgb
        write_result(details, details_path)
    del scanned
    # 仅重做结果可学习性诊断，精确复用既有 train/validation 行号。
    indexed_sample = sample.set_index("row_id", drop=False)
    parts = {
        name: indexed_sample.loc[pd.read_csv(out / f"data_preparation/{name}_row_ids.csv")["row_id"]].reset_index(drop=True)
        for name in ("train", "validation")
    }
    run_learnability_diagnostics(parts["train"], parts["validation"], out / "diagnostics", seed=args.seed)
    del sample, indexed_sample, parts
    manifest["diagnostic_models"] = ["logistic", "xgboost"]
    manifest["arguments"]["permutations"] = args.permutations
    manifest["arguments"]["c2st_sample"] = args.c2st_sample
    write_result(manifest, manifest_path)
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

    # 01. 为本次实验一次性创建总目录和固定的各阶段子目录。
    out = safe_output(
        args.output
        or ROOT / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"结果目录已有正式实验：{out}。仅更新诊断请使用 --diagnostics-only。"
        )
    out.mkdir(parents=True, exist_ok=True)
    data_out = out / "data_preparation"
    diagnostics_out = out / "diagnostics"
    models_out = out / "models"
    evaluation_out = out / "evaluation"
    for directory in (data_out, diagnostics_out, models_out, evaluation_out, out / "figures"):
        directory.mkdir(exist_ok=True)

    # 02. 准备建模样本、全量审计、来源指纹、诊断结果；
    sample, audit, source, scanned = prepare_sample(args)
    # 建模样本划分、重叠性检验
    parts, split_summary, feature_overlap = split_sample(sample, args.outcome, args.seed)
    
    # 分布图在运行阶段生成，Notebook 只读取已保存的图片。
    save_sample_feature_plot(sample, out / "figures/sample_features.png")
    del sample

    train, validation, test = (
            parts["train"],
            parts["validation"],
            parts["test"],
    )

    assignment_probability = float(train.treatment.mean())

    # 03. 保存上述结果

    for name, frame in parts.items():
        write_result(frame[["row_id"]], data_out / f"{name}_row_ids.csv")
    write_result(split_summary, data_out / "split_summary.csv")
    write_result(feature_overlap, data_out / "feature_overlap.json")
    write_result(audit, data_out / "audit.json")
    write_result(scanned["sample_report"], data_out / "sample_report.json")

    # 04.汇总ATE结果
    ate_full = pd.DataFrame([
        aggregate_ate(audit, outcome) for outcome in ("conversion", "visit")
    ])
    write_result(ate_full, diagnostics_out / "ate_full.csv")

    # 05. 执行 SRM、全量 SMD/KS 和 C2ST；复用扫描结果避免重复遍历。
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

    # 09. 保存可复现信息、生成报告，并标记实验完成。
    #-----------------------------------------------------------------------------------------------------------------------
    # 文件：manifest.json —— 本次实验的配置与运行记录。
    # 生成方式：
    #     run_pipeline.py 的 run() 组装内容，通过 write_result() 保存。
    #     finalize_run() 在报告生成成功后，将 status 更新为 complete。
    # 保存位置：当前实验结果目录
    # 主要内容：
    #     primary_outcome：分析的结果变量，例如 conversion 或 visit。
    #     sample_n、seed：实际建模样本量和随机种子。
    #     source：原始数据的来源、文件大小及 SHA256 指纹等。
    #     features：模型使用的特征名称。
    #     arguments：本次运行的完整命令行配置。
    #     selection：验证集冻结的策略选择、主预算和选择依据。
    #     models：各模型的参数与训练耗时。
    #     assignment_probability：评价使用的处理分配概率。
    #     指标口径与假设：评价目标、区间方法、指标版本及未包含的不确定性等。
    #     python、platform、versions：Python、操作系统及依赖库版本。
    #     status、elapsed_seconds：运行状态及写入时已累计的运行耗时。
    # 用途：
    #     记录这批实验结果的产生条件，供 Notebook 展示、结果追溯与实验复现使用。
    #     保存配置和说明；逐行样本、模型对象及详细指标分别保存在其他结果文件中。
    #-----------------------------------------------------------------------------------------------------------------------

    write_result(
        {
            "assignment_probability": assignment_probability,
            "probability_source": "使用训练集的处理组比例作为分配概率的代入估计",
            "ci_method": (
                "在处理组和对照组内分别进行百分位 Bootstrap 重抽样，"
                "各策略共用同一次重抽样；固定已训练模型的评分和处理分配概率，"
                "每次重抽样后重新选择排名前 k 的样本"
            ),
            "uncertainty_excluded": [
                "模型训练与模型选择带来的不确定性",
                "处理分配概率估计带来的不确定性",
                "无法识别的用户依赖关系",
            ],
            "estimand": (
                "公开样本中，按处理分配计算的增量结果"
                "（类似 ITT 口径），以所有人均不接受处理为比较基准"
            ),
            "cost_assumption": (
                "假设每次触达成本相同、每个结果事件的价值相同；"
                "预算表示触达人群比例，而非货币金额"
            ),
            "ranking_metric_version": (
                "v2：传统 Qini 保留仅含对照组的零值前缀；"
                "明确报告代入估计的 AUUC；"
                "不按完美排序的面积进行归一化"
            ),
            "diagnostic_models": ["logistic", "xgboost"],
            "primary_outcome": args.outcome,
            "selection": fitted["selection"],
            "models": fitted["model_metadata"],
            "assumptions": [
                "公开样本中，给定特征 X 的处理分配概率近似恒定",
                "样本行相互独立，满足一致性和个体间无干扰假设",
                "每次转化的价值相同，每次触达的成本相同",
            ],
            "status": "metrics_complete",  # 状态码：指标计算完成
            "source": source,
            "features": FEATURES,
            "seed": args.seed,
            "sample_n": sum(len(frame) for frame in parts.values()),
            "arguments": vars(args),
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "statsmodels",
                    "scikit-learn",
                    "xgboost",
                    "matplotlib",
                    "joblib",
                )
            },
            "elapsed_seconds": time.perf_counter() - started_at,
        },
        out / "manifest.json",
    )
    return finalize_run(out, policies, args.primary_budget, progress=log)


#-----------------------------------------------------------------------------------------------------------------------
# 函数：parse_args —— 解析并检查命令行配置。
# 输入：
#     argv：命令行参数列表；None 表示读取当前进程的参数。
# 输出：
#     args：解析后的命令行配置对象。
# 说明：结构化配置，就是把用户输入的命令行参数，按照预先定义的规则解析、转换并整理成一个具有明确属性的对象，供后续程序读取和使用。
#-----------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='运行完整 Criteo uplift 分析：数据准备、诊断、训练、验证集选择、测试集评价和报告。\n 原始 data 目录只读；运行示例：python run_pipeline.py --sample-size 1000000 --outcome conversion。'
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
        help='原位补齐审计/ATE尾概率说明，更新全量诊断、结果可学习性及报告；不重新划分或训练正式增量模型',
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
#     argv：（argument vector）命令行参数列表；None 表示读取当前进程的参数。
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
