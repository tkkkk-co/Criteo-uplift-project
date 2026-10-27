#!/usr/bin/env python3
# 加载可信实验目录中的冻结策略，对输入特征分块打分并按指定预算选择记录。
# 只输出原始行号、分数和选择标记；不训练模型、不读取结果标签，不执行真实广告投放。

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path

from criteo_uplift.runtime import configure_runtime

configure_runtime()

import joblib
import numpy as np
import pandas as pd

from criteo_uplift.data import FEATURES
from criteo_uplift.pipeline_io import safe_output

UPLIFT_POLICIES = {"s_logistic", "t_logistic", "t_xgboost", "x_xgboost"}
MAX_CHUNK_SIZE = 100_000


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_read_json —— 读取并验证冻结实验的 JSON 对象。
# 输入：
#     path：需要读取或检查的文件路径。
# 输出：
#     value：解析后的字典。
# 说明：读取并验证冻结实验的 JSON 对象。
#-----------------------------------------------------------------------------------------------------------------------
def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"冻结 run 缺少文件：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 run JSON：{path}；{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"run JSON 必须是对象：{path}")
    return value


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_load_frozen_policy —— 核对并加载冻结策略。
# 输入：
#     run：可信的冻结实验目录。
# 输出：
#     policy：冻结的策略名称。
#     seed：实验记录的随机种子。
#     model：已加载模型；随机策略时为 None。
# 说明：核对并加载冻结策略。
#-----------------------------------------------------------------------------------------------------------------------
def _load_frozen_policy(run):
    run = Path(run).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run 目录不存在：{run}")
    selection = _read_json((run / 'models' / 'selection.json'))
    manifest = _read_json(run / "manifest.json")
    for field in ("selected_policy", "frozen_before_test"):
        if field not in selection:
            raise ValueError(f"selection.json 缺少字段：{field}")
    if selection["frozen_before_test"] is not True:
        raise ValueError(
            "selection.json 未声明 frozen_before_test=true，不能作为冻结策略。"
        )
    for field in ("features", "seed", "selection"):
        if field not in manifest:
            raise ValueError(f"manifest.json 缺少字段：{field}")
    if manifest["features"] != FEATURES:
        raise ValueError("manifest.json 的 features 必须按顺序为 f0-f11。")
    policy = selection["selected_policy"]
    if policy not in {"random", "propensity", *UPLIFT_POLICIES}:
        raise ValueError(f"无法识别冻结策略 selected_policy={policy!r}。")
    saved_selection = manifest["selection"]
    if (
        not isinstance(saved_selection, dict)
        or saved_selection.get("selected_policy") != policy
        or saved_selection.get("frozen_before_test") is not True
    ):
        raise ValueError("manifest.json 与 selection.json 的冻结选择不一致。")
    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("manifest.json 的 seed 必须为非负整数。")
    model = None
    if policy != "random":
        model_name = "t_xgboost" if policy == "propensity" else policy
        path = (run / 'models' / f'{model_name}.joblib')
        if not path.is_file():
            raise FileNotFoundError(f"冻结策略 {policy} 所需模型不存在：{path}")
        try:
            model = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"无法载入可信本地模型 {path}：{exc}") from exc
        required_method = "predict_outcomes" if policy == "propensity" else "predict"
        if not callable(getattr(model, required_method, None)):
            raise ValueError(f"模型 {path.name} 缺少 {required_method}(X) 接口。")
        if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
            raise ValueError(f"模型 {path.name} 的训练特征数不是 12。")
    return policy, seed, model


#-----------------------------------------------------------------------------------------------------------------------
# 函数：_check_input_header —— 检查打分输入的特征列是否齐全且唯一。
# 输入：
#     path：需要读取或检查的文件路径。
# 输出：
#     返回值：None；列名不合法时抛出异常。
# 说明：检查打分输入的特征列是否齐全且唯一。
#-----------------------------------------------------------------------------------------------------------------------
def _check_input_header(path):
    if not path.is_file():
        raise FileNotFoundError(f"输入 CSV 不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    missing = [feature for feature in FEATURES if feature not in header]
    duplicated = [feature for feature in FEATURES if header.count(feature) > 1]
    if missing:
        raise ValueError(f"输入 CSV 缺少前置特征：{', '.join(missing)}")
    if duplicated:
        raise ValueError(f"输入 CSV 存在重复特征列：{', '.join(duplicated)}")


#-----------------------------------------------------------------------------------------------------------------------
# 函数：score_policy —— 使用冻结策略分块打分并执行全局预算排序。
# 输入：
#     run：可信的冻结实验目录。
#     input_path：只读输入文件路径。
#     output_path：输出文件路径。
#     budget：本次允许触达的人群比例。
#     chunk_size：每个数据块的行数上限。
# 输出：
#     结果字典：策略、种子、输入行数、预算、选中人数及输出路径。
# 说明：只使用处理前特征，不重新训练或选择策略；输出原始行号、分数和选择标记。
#-----------------------------------------------------------------------------------------------------------------------
def score_policy(run, input_path, output_path, budget=0.2, chunk_size=MAX_CHUNK_SIZE):
    if isinstance(budget, bool):
        raise ValueError("budget 必须在 [0, 1] 内。")
    try:
        budget_value = Decimal(str(budget))
    except Exception as exc:
        raise ValueError("budget 必须是 [0, 1] 内的数值。") from exc
    if not budget_value.is_finite() or not Decimal(0) <= budget_value <= Decimal(1):
        raise ValueError("budget 必须在 [0, 1] 内。")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or not 1 <= chunk_size <= MAX_CHUNK_SIZE
    ):
        raise ValueError(f"chunk_size 必须为 1 到 {MAX_CHUNK_SIZE} 的整数。")
    input_path = Path(input_path).resolve()
    requested_output = Path(output_path)
    output_path = safe_output(requested_output)
    if output_path == input_path:
        raise ValueError("输出路径不能等于输入 CSV；输入文件只读。")
    if requested_output.is_symlink() or output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{requested_output}")
    _check_input_header(input_path)
    policy, seed, model = _load_frozen_policy(run)
    score_seed = np.random.SeedSequence(seed).spawn(1)[0]
    score_rng = np.random.default_rng(score_seed)
    # 与 evaluation._rank 保持一致，支持相同原始行序下复现选择结果。
    # 随机策略的分数使用上方单独派生的随机流。
    tie_rng = np.random.default_rng(seed)
    scores = []
    offset = 0
    with pd.read_csv(
        input_path,
        usecols=FEATURES,
        dtype={feature: "float32" for feature in FEATURES},
        chunksize=chunk_size,
        encoding="utf-8-sig",
    ) as reader:
        for chunk in reader:
            # 显式指定列顺序并只读取特征，排除结果、曝光和处理标签。
            X = chunk[FEATURES]
            if len(X) == 0:
                continue
            if not np.isfinite(X.to_numpy()).all():
                raise ValueError(f"输入行偏移 {offset} 起包含缺失或非有限前置特征。")
            if policy == "random":
                score = score_rng.random(len(X))
            elif policy == "propensity":
                # 响应概率基线预测处理状态下的结果概率，区别于处理分配概率。
                score = model.predict_outcomes(X)[1]
            else:
                score = model.predict(X)
            score = np.asarray(score, dtype=np.float64)
            if score.shape != (len(X),) or not np.isfinite(score).all():
                raise ValueError(f"模型在输入行偏移 {offset} 返回非法 score。")
            scores.append(score)
            offset += len(X)
    if offset == 0:
        raise ValueError("输入 CSV 没有数据行。")
    scores = np.concatenate(scores)
    # 使用 Decimal，避免 0.58 乘以 100 因二进制误差被向下取整成 57。
    selected_n = int((budget_value * offset).to_integral_value(rounding=ROUND_FLOOR))
    selected = np.zeros(offset, dtype=np.int8)
    if selected_n:
        order = np.lexsort((tie_rng.random(offset), -scores))
        selected[order[:selected_n]] = 1
        del order
    # 使用独占创建，防止预检查后新出现的同名文件被覆盖。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        for start in range(0, offset, chunk_size):
            stop = min(start + chunk_size, offset)
            pd.DataFrame(
                {
                    "source_row_id": np.arange(start, stop),
                    "score": scores[start:stop],
                    "selected": selected[start:stop],
                }
            ).to_csv(handle, index=False, header=start == 0)
    return {
        "selected_policy": policy,
        "seed": seed,
        "n_rows": offset,
        "budget": float(budget_value),
        "selected_n": selected_n,
        "output": str(output_path),
    }


#-----------------------------------------------------------------------------------------------------------------------
# 函数：main —— 解析命令行并执行离线策略打分。
# 输入：
#     argv：命令行参数列表；None 表示读取当前进程的参数。
# 输出：
#     result：本次打分结果的概要字典。
# 说明：解析命令行并执行离线策略打分。
#-----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='加载可信实验目录中的冻结策略，对输入特征分块打分并按指定预算选择记录。\n只输出原始行号、分数和选择标记；不训练模型、不读取结果标签，不执行真实广告投放。', formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="已有可信冻结 run 目录")
    parser.add_argument(
        "--input", type=Path, required=True, help="包含 f0-f11 的输入 CSV，只读"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="新的结果 CSV，拒绝覆盖"
    )
    parser.add_argument(
        "--budget", type=float, default=0.2, help="覆盖比例 [0,1]，默认 0.2"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=MAX_CHUNK_SIZE,
        help="每批特征行数，最多 100000",
    )
    args = parser.parse_args(argv)
    try:
        result = score_policy(
            args.run, args.input, args.output, args.budget, args.chunk_size
        )
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
