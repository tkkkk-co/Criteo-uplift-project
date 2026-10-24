#!/usr/bin/env python3
"""Run the end-to-end Criteo uplift analysis pipeline.

Example: python run_pipeline.py --sample-size 1000000 --outcome conversion
The pipeline writes artifacts only; the original data/ directory is read-only.
"""

# Configure numerical-library threads before importing NumPy, pandas, or models.
from criteo_uplift.runtime import configure_runtime

configure_runtime()

import argparse
import time
from pathlib import Path

from criteo_uplift.data import (
    ROOT,
    DEFAULT_DATA,
    safe_output,
    split_sample,
)
from criteo_uplift.diagnostics import update_experiment_diagnostics
from criteo_uplift.evaluation import aggregate_ate_from_audit as aggregate_ate
from criteo_uplift.preparation import prepare_sample as _prepare_sample
from criteo_uplift.training import (
    fit_and_select_models,
    run_learnability_diagnostics,
)
from criteo_uplift.test_evaluation import evaluate_test_set
from criteo_uplift.pipeline_io import (
    finalize_run,
    update_diagnostics_only,
    write_audit_artifacts,
    write_feature_overlap,
    write_run_manifest,
    write_split_artifacts,
)


# ----------------------------------------------------------------------------
# Function: log — print a timestamped progress message.
# Inputs:
#     message: text describing the current pipeline step.
# Outputs:
#     None; writes one flushed line to standard output.
# ----------------------------------------------------------------------------
def log(message):
    """Print a timestamped progress message immediately."""
    import time as _time

    print(_time.strftime("%H:%M:%S"), message, flush=True)


# ----------------------------------------------------------------------------
# Function: prepare_sample — compatibility entry point for sample preparation.
# Inputs:
#     args: CLI-style configuration for source, sample, seed, and diagnostics.
# Outputs:
#     Same four-item tuple returned by criteo_uplift.preparation.prepare_sample.
# Notes:
#     Keeps existing imports/tests working while routing implementation to the
#     preparation module. ROOT and log are passed explicitly for easy testing.
# ----------------------------------------------------------------------------
def prepare_sample(args):
    """Prepare or reuse the modeling sample and same-scan diagnostic data."""
    return _prepare_sample(args, root=ROOT, progress=log)


# ----------------------------------------------------------------------------
# Function: run — orchestrate the complete analysis in notebook reading order.
# Inputs:
#     args: parsed CLI configuration (or a compatible namespace in tests).
# Outputs:
#     Path to the completed run directory.
# Notes:
#     The implementation of each stage lives in its responsibility-specific
#     module; this function keeps the end-to-end call sequence visible.
# ----------------------------------------------------------------------------
def run(args):
    """Execute data preparation, diagnostics, model selection, and test evaluation.

    Args:
        args: namespace with data, sample_size, outcome, seed, primary_budget,
            permutations, c2st_sample, bootstrap, trees, and optional output.

    Returns:
        Path to the completed experiment output directory.
    """
    started_at = time.perf_counter()

    # 01. Create a fresh output directory for this run.
    out = safe_output(
        args.output
        or ROOT / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"结果目录已有正式实验：{out}。仅更新诊断请使用 --diagnostics-only。"
        )
    out.mkdir(parents=True, exist_ok=True)

    # 02. Prepare the sample, full-source audit, fingerprint, and same-scan diagnostics.
    sample, audit, source, scanned = prepare_sample(args)
    parts = split_sample(sample, args.outcome, args.seed)
    del sample

    # 03. Persist split IDs/summary and check identical feature vectors across splits.
    write_split_artifacts(parts, out)
    train, validation, test = (
        parts["train"],
        parts["validation"],
        parts["test"],
    )
    assignment_probability = float(train.treatment.mean())
    write_feature_overlap(parts, out)

    # 04. Save the full-source audit and aggregate ATE results.
    write_audit_artifacts(audit, out)

    # 05. Run SRM, full-source SMD/KS, and C2ST; reuse scan outputs to avoid a second scan.
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

    # 06. Check whether pre-treatment features predict the observed outcomes.
    run_learnability_diagnostics(
        train, validation, out, seed=args.seed
    )

    # 07. Fit candidates on train and freeze the policy using validation only.
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

    # 08. Evaluate the frozen models once on test and persist test artifacts.
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

    # 09. Save reproducibility metadata, render the report, and mark the run complete.
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


def parse_args(argv=None):
    """Parse CLI options.

    Args:
        argv: optional argument list; None reads the process command line.

    Returns:
        argparse.Namespace with data, modeling, diagnostics, and output options.
    """
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


# ----------------------------------------------------------------------------
# Function: main — dispatch the full pipeline or diagnostics-only refresh.
# Inputs:
#     argv: optional command-line argument list.
# Outputs:
#     None; runs the requested workflow and writes artifacts.
# ----------------------------------------------------------------------------
def main(argv=None):
    """Parse CLI options and run the selected workflow."""
    args = parse_args(argv)
    if args.diagnostics_only:
        update_diagnostics_only(args, root=ROOT, progress=log)
    else:
        run(args)


if __name__ == "__main__":
    main()
