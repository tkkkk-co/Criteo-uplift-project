# 只使用开发数据研究训练规模和划分方式的敏感性，排除原测试集。
# 重复开发验证之间存在重叠，结果离散程度属于描述性统计，不是新的独立测试证据。

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import FEATURES, POLICY_NAMES
from .pipeline_io import safe_output, write_result
from .metrics import RANKING_METRIC_DESCRIPTION, RANKING_METRICS, evaluate_ranking
from .training_validation import _fit_candidates, policy_scores
from .diagnostics import diagnose_overlap
from .input_checks import check_row_ids, check_binary_vector

DEFAULT_SEEDS = (17, 29, 83)
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)


def _validate_fractions(fractions):
    #-------------
    # 函数名：_validate_fractions
    # 功能：校验训练样本比例为严格递增、唯一且位于 (0,1] 的序列。
    # 输入参数：
    #     - fractions：待检查的训练样本比例序列。
    # 输出参数：
    #     - 转换为 float 的比例元组。
    #-------------
    fractions = tuple(float(f) for f in fractions)
    if (
        not fractions
        or not np.isfinite(fractions).all()
        or any(f <= 0 or f > 1 for f in fractions)
        or tuple(sorted(set(fractions))) != fractions
    ):
        raise ValueError(
            "fractions 必须为 (0, 1] 内严格递增且不重复的数值"
        )
    return fractions


def _validate_seeds(seeds):
    #-------------
    # 函数名：_validate_seeds
    # 功能：校验随机种子为非负、唯一且位于 NumPy 允许范围内的整数。
    # 输入参数：
    #     - seeds：待检查的随机种子序列。
    # 输出参数：
    #     - 转换为 Python int 的种子元组。
    #-------------
    seeds = tuple(seeds)
    if (
        not seeds
        or any(
            not isinstance(s, (int, np.integer))
            or isinstance(s, bool)
            or s < 0
            or s >= 2**32
            for s in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds 必须为 [0, 2**32) 内不重复的整数")
    return tuple(int(s) for s in seeds)


def feature_groups(frame):
    #-------------
    # 函数名：feature_groups
    # 功能：将 f0–f11 转为 float32 后计算每行匿名特征向量哈希，用于特征分组隔离检查。
    # 输入参数：
    #     - frame：包含 f0–f11 的样本 DataFrame。
    # 输出参数：
    #     - 每行对应一个匿名特征组的 uint64 哈希数组。
    #-------------
    return pd.util.hash_pandas_object(
        frame[FEATURES].astype(np.float32), index=False
    ).to_numpy(dtype=np.uint64)


def nested_stratified_indices(t, y, fractions=DEFAULT_FRACTIONS, seed=17):
    #-------------
    # 函数名：nested_stratified_indices
    # 功能：在 treatment × outcome 分层内生成嵌套、无放回的训练样本位置索引。
    # 输入参数：
    #     - t：处理分配二元向量。
    #     - y：结果变量二元向量。
    #     - fractions：严格递增的训练样本比例序列。
    #     - seed：分层抽样随机种子。
    # 输出参数：
    #     - 字典：以训练比例为键、对应样本位置索引数组为值。
    #-------------
    check_binary_vector(t, "t", require_both=True)
    check_binary_vector(y, "y", len(t))
    t, y = np.asarray(t, dtype=np.int8), np.asarray(y, dtype=np.int8)
    fractions = _validate_fractions(fractions)
    _validate_seeds((seed,))
    rng = np.random.default_rng(seed)
    strata = t * 2 + y
    orders = [rng.permutation(np.flatnonzero(strata == s)) for s in np.unique(strata)]
    result = {}
    for fraction in fractions:
        indices = np.concatenate(
            [order[: int(np.floor(fraction * len(order)))] for order in orders]
        )
        # 排序只恢复来源行序；入选成员仍由随机嵌套抽样决定。
        indices = np.sort(indices)
        if len(indices) == 0 or len(np.unique(t[indices])) != 2:
            raise ValueError(
                f"比例 {fraction} 下保留的行数不足或缺少某个处理臂"
            )
        result[fraction] = indices
    return result


def development_splits(frame, seed=17, outcome="conversion"):
    #-------------
    # 函数名：development_splits
    # 功能：为 Stability 开发样本构造行级 T×Y 分层划分和匿名特征分组划分，并检查两种划分无泄漏。
    # 输入参数：
    #     - frame：已排除原 Test 的开发样本 DataFrame。
    #     - seed：本次开发划分随机种子。
    #     - outcome：用于分层和模型训练的二元结果列名。
    # 输出参数：
    #     - 字典：row_stratified 和 feature_group 两种划分各自的训练/验证位置索引二元组。
    #-------------
    _validate_seeds((seed,))
    t, y = frame.treatment.to_numpy(), frame[outcome].to_numpy()
    strata = t * 2 + y
    counts = np.unique(strata, return_counts=True)[1]
    if counts.min() < 2:
        raise ValueError(
            "每个已观测的 T × Y 分层至少需要两条开发集记录"
        )
    positions = np.arange(len(frame))
    train, validation = train_test_split(
        positions, test_size=0.25, stratify=strata, random_state=seed
    )
    groups = feature_groups(frame)
    if len(np.unique(groups)) < 2:
        raise ValueError("特征分组敏感性分析至少需要两个特征组")
    group_train, group_validation = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            positions, groups=groups
        )
    )
    result = {
        "row_stratified": (train, validation),
        "feature_group": (group_train, group_validation),
    }
    for name, (training, holdout) in result.items():
        check_binary_vector(t[training], '训练处理分配', require_both=True)
        check_binary_vector(t[holdout], '验证处理分配', require_both=True)
        if np.intersect1d(training, holdout).size:
            raise AssertionError(f"{name} has overlapping row positions")
    if np.intersect1d(groups[group_train], groups[group_validation]).size:
        raise AssertionError("feature groups overlap across the group split")
    return result


def _counts(frame, outcome, prefix):
    #-------------
    # 函数名：_counts
    # 功能：汇总指定 DataFrame 中 Control/Treatment 人数及对应结果正例数，并添加字段前缀。
    # 输入参数：
    #     - frame：需要统计的训练或验证 DataFrame。
    #     - outcome：统计的二元结果列名。
    #     - prefix：输出字段使用的前缀，例如 train 或 validation。
    # 输出参数：
    #     - 人数和正例数统计字典。
    #-------------
    t, y = frame.treatment.to_numpy(), frame[outcome].to_numpy()
    return {
        f"{prefix}_n": len(frame),
        f"{prefix}_control_n": int((t == 0).sum()),
        f"{prefix}_treatment_n": int((t == 1).sum()),
        f"{prefix}_control_events": int(y[t == 0].sum()),
        f"{prefix}_treatment_events": int(y[t == 1].sum()),
    }


def _summarize(longmetrics):
    #-------------
    # 函数名：_summarize
    # 功能：按划分方式、训练比例和策略汇总 Stability 指标的均值、标准差、极值及 Validation 选择频率。
    # 输入参数：
    #     - longmetrics：每次开发实验逐策略记录的长表 DataFrame。
    # 输出参数：
    #     - 二元组：(聚合指标 DataFrame、策略选择频率 DataFrame)。
    #-------------
    dimensions = ["split_method", "training_fraction", "policy"]
    quantities = [
        "gain_ht_per_100k", "vs_propensity_per_100k", "bootstrap_mean",
        *RANKING_METRICS, "fit_seconds", "train_n", "train_control_events",
        "validation_control_events",
    ]
    aggregate = longmetrics.groupby(dimensions, sort=False)[quantities].agg(
        ["mean", "std", "min", "max"]
    )
    aggregate.columns = [
        f"{quantity}_{statistic}" for quantity, statistic in aggregate.columns
    ]
    aggregate = aggregate.reset_index()
    frequencies = (
        longmetrics.groupby(dimensions, sort=False)
        .agg(selected_count=("selected_on_validation", "sum"), runs=("seed", "nunique"))
        .reset_index()
    )
    frequencies["selected_fraction"] = frequencies.selected_count / frequencies.runs
    return aggregate, frequencies


def run_stability(
    development,
    test_row_ids,
    output,
    *,
    seeds=DEFAULT_SEEDS,
    fractions=DEFAULT_FRACTIONS,
    trees=150,
    primary_budget=0.2,
    primary_metric="gain_ht",
    n_bootstrap=200,
    outcome="conversion",
):
    #-------------
    # 函数名：run_stability
    # 功能：在排除原 Test 的开发数据上运行多随机种子、训练规模和划分方式敏感性分析，并保存协议、明细和汇总。
    # 输入参数：
    #     - development：本次 Train 和 Validation。
    #     - test_row_ids：本次独立 Test 的来源行号。
    #     - output：本次 Stability 新输出目录。
    #     - seeds：重复开发实验使用的随机种子序列。
    #     - fractions：行分层划分下使用的嵌套训练比例序列。
    #     - trees：Stability 中 XGBoost 模型的树数量。
    #     - primary_budget：Gain 主指标的选型覆盖率。
    #     - primary_metric：与正式 Validation 一致的主指标。
    #     - n_bootstrap：各开发实验使用的重抽样次数。
    #     - outcome：模型训练和指标计算使用的结果列名。
    # 输出参数：
    #     - 完成后的 Stability 输出目录 pathlib.Path；同时写入 protocol、逐次指标、划分审计和汇总文件。
    #-------------
    from .config import configure_runtime

    configure_runtime()
    seeds, fractions = _validate_seeds(seeds), _validate_fractions(fractions)
    if not isinstance(trees, (int, np.integer)) or isinstance(trees, bool) or trees < 1:
        raise ValueError("trees 必须为正整数")
    if not np.isfinite(primary_budget) or not 0 < primary_budget <= 1:
        raise ValueError("primary_budget 必须属于 (0, 1]")
    if primary_metric not in RANKING_METRICS or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("Stability 需要有效主指标和至少两次 Bootstrap")
    output = safe_output(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output already exists and is nonempty: {output}")
    started = time.perf_counter()
    check_row_ids(development.row_id, "Development", require_integer=True)
    check_row_ids(test_row_ids, "Test", require_integer=True)
    if np.isin(development.row_id, test_row_ids).any():
        raise ValueError("Development 与 Test 的来源行号必须互斥")
    # 在拟合任何模型前，先准备并校验全部划分。
    splits = {seed: development_splits(development, seed, outcome) for seed in seeds}
    protocol = {
        "study": "development_training_size_and_split_sensitivity",
        "status": "frozen_before_model_fits",
        "seeds": seeds,
        "training_fractions": fractions,
        "primary_budget": primary_budget,
        "outcome": outcome,
        "features": FEATURES,
        "candidates": POLICY_NAMES,
        "model_seed": "equal to the current split seed; all other model parameters are fixed",
        "row_split": "75/25 T x Y stratified development split; all configured nested fractions",
        "training_sampling": "within each T x Y stratum, one random permutation; floor(fraction*n_stratum), no replacement, no event oversampling",
        "group_split": "GroupShuffleSplit 75/25 distinct 64-bit hashes of 12 float32 features; full training fraction only; row counts may differ",
        "group_interpretation": "identical anonymous feature vectors are not established user identities; no automatic deduplication; hash collisions theoretically possible",
        "assignment_probability": "plug-in treatment fraction of the full training partition, shared across its nested fractions; validation outcomes do not estimate it",
        "primary_metric": primary_metric,
        "n_bootstrap": n_bootstrap,
        "selection_statistic": "bootstrap_mean",
        "ranking_metric_definition_from_loaded_function": RANKING_METRIC_DESCRIPTION,
        "response_baseline": "response_logistic and response_xgboost estimate P(Y=1|X,T=1), reused from corresponding T-Learners; not P(T=1|X)",
        "selection": "one development winner by bootstrap mean per seed/split/fraction; stable ties follow candidate order",
        "uncertainty": "descriptive spread across overlapping development runs; each run selects by its own fixed-score bootstrap mean",
        "limitations": [
            "The group holdout changes feature support; its values are sensitivity results, not a matched causal comparison of splitting methods.",
            "Rare-event counts and performance associations do not establish a causal explanation for model differences.",
            "Equal contact cost and conversion value are assumptions; public-sample assignment effects are not online revenue.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_result(protocol, output / "protocol.json")
    records, audits = [], []
    for seed in seeds:
        for method, (train_positions, val_positions) in splits[seed].items():
            full_train = development.iloc[train_positions].reset_index(drop=True)
            validation = development.iloc[val_positions].reset_index(drop=True)
            e = float(full_train.treatment.mean())
            chosen_fractions = fractions if method == "row_stratified" else (1.0,)
            subsets = nested_stratified_indices(
                full_train.treatment, full_train[outcome], chosen_fractions, seed
            )
            val_groups = feature_groups(validation)
            for fraction, subset in subsets.items():
                train = full_train.iloc[subset]
                print(
                    f"stability seed={seed} split={method} fraction={fraction:g} train={len(train):,} validation={len(validation):,}",
                    flush=True,
                )
                train_groups = feature_groups(train)
                group_overlap = int(np.intersect1d(train_groups, val_groups).size)
                audit = {
                    "seed": seed,
                    "split_method": method,
                    "training_fraction": fraction,
                    **_counts(train, outcome, "train"),
                    **_counts(validation, outcome, "validation"),
                    "full_train_n": len(full_train),
                    "validation_fraction_of_development": len(validation)
                    / len(development),
                    "assignment_probability": e,
                    "row_overlap": int(
                        np.intersect1d(train.row_id, validation.row_id).size
                    ),
                    "train_test_overlap": int(
                        np.isin(train.row_id, test_row_ids).sum()
                    ),
                    "validation_test_overlap": int(
                        np.isin(validation.row_id, test_row_ids).sum()
                    ),
                    "feature_group_overlap": group_overlap,
                    "validation_fraction_matching_train_features": float(
                        np.isin(val_groups, train_groups).mean()
                    ),
                }
                if any(
                    audit[name]
                    for name in (
                        "row_overlap",
                        "train_test_overlap",
                        "validation_test_overlap",
                    )
                ):
                    raise AssertionError("forbidden row leakage detected")
                if method == "feature_group" and group_overlap:
                    raise AssertionError("feature group leakage detected")
                audits.append(audit)
                # 每个训练子集独立检查 Overlap，所有候选共享这次诊断。
                coverage = diagnose_overlap(train[FEATURES], train.treatment)['summary']
                audit['overlap_status'] = coverage['status']
                audit['overlap_weak_bins'] = coverage['weak_bins']

                results = _fit_candidates(train, validation, outcome, seed, trees, progress=lambda _: None)
                score_dict = policy_scores(results)
                fit_seconds = {name: result['fit_seconds'] for name, result in results.items()}
                del results
                evaluated = evaluate_ranking(
                    validation[outcome], validation.treatment, score_dict, e,
                    budgets=[primary_budget], primary_metric=primary_metric, primary_budget=primary_budget,
                    n_bootstrap=n_bootstrap, seed=seed, include_curves=False,
                )
                means = evaluated["bootstrap"]["summary"].query("policy != 'random'").set_index("policy")
                if means.bootstrap_mean.isna().all():
                    raise ValueError("Stability 主指标没有可用的 Bootstrap 均值")
                winner = means.bootstrap_mean.idxmax()
                points = evaluated["metrics"]
                propensity_ht = points.loc[points.policy.eq("response_xgboost"), "gain_ht_per_100k"].iloc[0]
                for row in points.to_dict("records"):
                    name = row["policy"]
                    records.append({
                        **{key: audit[key] for key in (
                            "seed", "split_method", "training_fraction", "assignment_probability",
                            "train_n", "train_control_n", "train_treatment_n", "train_control_events",
                            "train_treatment_events", "validation_n", "validation_control_n",
                            "validation_treatment_n", "validation_control_events",
                            "validation_treatment_events",
                        )},
                        **row,
                        "vs_propensity_per_100k": row["gain_ht_per_100k"] - propensity_ht,
                        "bootstrap_mean": float(means.loc[name, "bootstrap_mean"]),
                        "fit_seconds": fit_seconds.get(name, 0.0),
                        "fit_source": "t_xgboost_reused" if name == "response_xgboost" else name,
                        "selected_on_validation": name == winner,
                    })
                # 进度文件只保存在本次新目录中；预先冻结的分析配置保持不变。
                write_result(pd.DataFrame(records), output / "longmetrics.csv")
                write_result(pd.DataFrame(audits), output / "split_audit.csv")
    longmetrics = pd.DataFrame(records)
    aggregate, frequencies = _summarize(longmetrics)
    write_result(aggregate, output / "aggregate.csv")
    write_result(frequencies, output / "selection_frequency.csv")
    report = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "experiments": len(audits),
        "fitted_models": len(audits) * 6,
        "development_n": len(development),
        "test_n_excluded": len(test_row_ids),
        "all_test_row_overlaps_zero": all(
            a["train_test_overlap"] == a["validation_test_overlap"] == 0
            for a in audits
        ),
        "all_group_split_feature_overlaps_zero": all(
            a["feature_group_overlap"] == 0
            for a in audits
            if a["split_method"] == "feature_group"
        ),
        "interpretation": "Development sensitivity only. Report training event counts, paired per-run differences and descriptive spread. Do not describe a validation winner as an independently verified improvement.",
        "notes": protocol["limitations"],
        "ranking_metric_definition_from_loaded_function": protocol[
            "ranking_metric_definition_from_loaded_function"
        ],
        "selection_frequency": frequencies.to_dict("records"),
        "outputs": [
            "protocol.json",
            "longmetrics.csv",
            "split_audit.csv",
            "aggregate.csv",
            "selection_frequency.csv",
            "report.json",
        ],
        "models_saved": False,
        "original_test_re_evaluated": False,
    }
    write_result(report, output / "report.json")
    return output
