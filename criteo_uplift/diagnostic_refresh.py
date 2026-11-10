"""已有实验的诊断维护流程；统计计算统一调用 diagnostics.py。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preparation import ROOT
from .diagnostics import update_experiment_diagnostics, run_learnability_diagnostics
from .evaluation import aggregate_ate_from_audit
from .pipeline_io import safe_output, write_result
from .preparation import prepare_sample
from .reporting import build_report


def update_diagnostics_only(args, *, root=ROOT, progress=print):
    """在原实验上更新诊断和报告，复用已保存的划分与正式模型。"""
    out = safe_output(
        args.output
        or Path(root) / f"artifacts/{args.outcome}_{args.sample_size}_seed{args.seed}"
    )
    manifest_path = out / "manifest.json"
    report_path = out / "report.html"
    audit = json.loads((out / "data_preparation/audit.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.seed != manifest["seed"]:
        raise ValueError("诊断 seed 必须与当前实验一致")
    # 缓存缺少审计字段时，prepare_sample 按正常失效规则在同次扫描中重建。
    saved_sample_report = json.loads((out / "data_preparation/sample_report.json").read_text())
    prepare_args = argparse.Namespace(**{**vars(args), "sample_size": saved_sample_report["sample_size"]})
    sample, refreshed_audit, source, scanned = prepare_sample(prepare_args, root=root, progress=progress)
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
        result = aggregate_ate_from_audit(audit, row["outcome"])
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
    build_report(manifest_path, report_path, diagnostics_only=True)
    progress(f"已原位更新诊断：{out}")
    return out
