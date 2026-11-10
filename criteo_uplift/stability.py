# 只使用开发数据研究训练规模和划分方式的敏感性，排除原测试集。
# Notebook 06 的稳定性分析实现；它是开发期敏感性研究，不是新的独立 test 确认。
# 重复开发验证之间存在重叠，结果离散程度属于描述性统计，不是新的独立测试证据。

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import FEATURES
from .pipeline_io import fingerprint
from .pipeline_io import load_sample, safe_output, write_result
from .metrics import RANKING_METRIC_DESCRIPTION, ranking_metrics
from .evaluation import evaluate_policies
from .models import make_uplift_model, predict_policy_scores
from .training_validation import MODEL_CANDIDATES
from .input_checks import check_row_ids, check_binary_vector, check_feature_matrix

POLICY_NAMES = ("random", *(c["name"] for c in MODEL_CANDIDATES), "propensity")
DEFAULT_SEEDS = (17, 29, 83)
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)


def row_ids_sha256(ids):
    """
    函数名：row_ids_sha256
    功能：对来源 row_id 排序后按固定 int64 字节编码计算 SHA256，用于记录样本行集合指纹。
    输入参数：
        - ids：来源 row_id 向量。
    输出参数：
        - 十六进制 SHA256 字符串。
    """
    return sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def _checked_ids(values, label):
    """
    函数名：_checked_ids
    功能：校验来源行号唯一、非空且为非负整数，再转换为 int64 数组。
    输入参数：
        - values：待校验的来源 row_id 向量。
        - label：用于错误信息的来源标识名称。
    输出参数：
        - 通过校验的 int64 row_id 数组。
    """
    check_row_ids(values, label, require_integer=True)
    return np.asarray(values, dtype=np.int64)


def _binary_vectors(t, y):
    """
    函数名：_binary_vectors
    功能：同时校验处理分配和结果变量为合法二元向量，并转换为 int8 数组。
    输入参数：
        - t：0/1 处理分配向量，要求两组均存在。
        - y：与 t 等长的 0/1 结果向量。
    输出参数：
        - 二元组：(int8 处理数组, int8 结果数组)。
    """
    check_binary_vector(t, "t", require_both=True)
    check_binary_vector(y, "y", len(t))
    return np.asarray(t, dtype=np.int8), np.asarray(y, dtype=np.int8)


def _validate_fractions(fractions):
    """
    函数名：_validate_fractions
    功能：校验训练样本比例为严格递增、唯一且位于 (0,1] 的序列。
    输入参数：
        - fractions：待检查的训练样本比例序列。
    输出参数：
        - 转换为 float 的比例元组。
    """
    fractions = tuple(float(f) for f in fractions)
    if (
        not fractions
        or not np.isfinite(fractions).all()
        or any(f <= 0 or f > 1 for f in fractions)
        or tuple(sorted(set(fractions))) != fractions
    ):
        raise ValueError(
            "fractions must be strictly increasing unique values in (0, 1]"
        )
    return fractions


def _validate_seeds(seeds):
    """
    函数名：_validate_seeds
    功能：校验随机种子为非负、唯一且位于 NumPy 允许范围内的整数。
    输入参数：
        - seeds：待检查的随机种子序列。
    输出参数：
        - 转换为 Python int 的种子元组。
    """
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
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    return tuple(int(s) for s in seeds)


def feature_groups(frame):
    """
    函数名：feature_groups
    功能：将 f0–f11 转为 float32 后计算每行匿名特征向量哈希，用于特征分组隔离检查。
    输入参数：
        - frame：包含 f0–f11 的样本 DataFrame。
    输出参数：
        - 每行对应一个匿名特征组的 uint64 哈希数组。
    """
    return pd.util.hash_pandas_object(
        frame[FEATURES].astype(np.float32), index=False
    ).to_numpy(dtype=np.uint64)


def load_development_sample(cache, base_run, outcome="conversion"):
    """
    函数名：load_development_sample
    功能：读取样本缓存和原实验清单，核对来源及 Test 行号，并排除原 Test 得到 Stability 开发样本。
    输入参数：
        - cache：建模样本 NPZ 缓存路径。
        - base_run：已有完整实验的输出目录。
        - outcome：Stability 使用的结果列名，须与基准实验一致。
    输出参数：
        - 三元组：(排除旧 Test 后的 development DataFrame、旧 Test row_id 数组、输入来源及行集合 provenance 字典)。
    """
    cache, base_run = Path(cache).resolve(), Path(base_run).resolve()
    if outcome not in ("conversion", "visit"):
        raise ValueError("outcome must be conversion or visit")
    manifest_path = base_run / "manifest.json"
    test_path = (base_run / 'data_preparation' / 'test_row_ids.csv')
    metadata_path = cache.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if manifest.get("status") not in ("complete", "models_frozen") or (
        manifest.get("status") == "models_frozen" and not manifest.get("selection", {}).get("frozen_before_test")
    ):
        raise ValueError("the base run must have a complete manifest")
    source_sha = manifest.get("source", {}).get("sha256")
    if not source_sha or source_sha != metadata.get("source", {}).get("sha256"):
        raise ValueError("cache metadata and base run source SHA256 disagree")
    if manifest.get("primary_outcome") != outcome:
        raise ValueError("study outcome must match the existing base run")
    sample = load_sample(cache)
    ids = _checked_ids(sample.row_id.to_numpy(), "cache")
    test_table = pd.read_csv(test_path)
    if list(test_table.columns) != ["row_id"]:
        raise ValueError("test_row_ids.csv must contain exactly the row_id column")
    old_test_ids = _checked_ids(test_table.row_id.to_numpy(), "old test")
    if not np.isin(old_test_ids, ids).all():
        raise ValueError("old test row IDs are not all present in the supplied cache")
    if manifest.get("sample_n") != len(sample):
        raise ValueError("cache row count disagrees with base manifest")
    requested = metadata.get("sample_size")
    source_rows = metadata.get("source_n")
    if (
        isinstance(requested, bool) or not isinstance(requested, int) or requested < 0
        or isinstance(source_rows, bool) or not isinstance(source_rows, int)
        or source_rows < len(sample) or metadata.get("sample_n") != len(sample)
    ):
        raise ValueError("cache row count disagrees with cache metadata")
    expected_n = source_rows if requested == 0 else min(requested, source_rows)
    if len(sample) != expected_n:
        raise ValueError("cache row count disagrees with cache metadata")
    check_feature_matrix(sample[FEATURES], n_features=len(FEATURES))
    _binary_vectors(sample.treatment, sample[outcome])
    development = sample.loc[~sample.row_id.isin(old_test_ids)].reset_index(drop=True)
    _binary_vectors(development.treatment, development[outcome])
    provenance = {
        "cache": fingerprint(cache),
        "cache_metadata": fingerprint(metadata_path),
        "base_manifest": fingerprint(manifest_path),
        "old_test_id_file": fingerprint(test_path),
        "raw_source_sha256_from_existing_metadata": source_sha,
        "sample_n": len(sample),
        "old_test_n": len(old_test_ids),
        "development_n": len(development),
        "development_row_ids_sha256": row_ids_sha256(development.row_id),
        "old_test_row_ids_sha256": row_ids_sha256(old_test_ids),
        "development_old_test_overlap": int(
            np.isin(development.row_id, old_test_ids).sum()
        ),
        "row_ids_sha256_encoding": "sorted little-endian signed int64, no header",
    }
    return development, old_test_ids, provenance


def nested_stratified_indices(t, y, fractions=DEFAULT_FRACTIONS, seed=17):
    """
    函数名：nested_stratified_indices
    功能：在 treatment × outcome 分层内生成嵌套、无放回的训练样本位置索引。
    输入参数：
        - t：处理分配二元向量。
        - y：结果变量二元向量。
        - fractions：严格递增的训练样本比例序列。
        - seed：分层抽样随机种子。
    输出参数：
        - 字典：以训练比例为键、对应样本位置索引数组为值。
    """
    t, y = _binary_vectors(t, y)
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
                f"fraction {fraction} leaves too few rows or a missing treatment arm"
            )
        result[fraction] = indices
    return result


def development_splits(frame, seed=17, outcome="conversion"):
    """
    函数名：development_splits
    功能：为 Stability 开发样本构造行级 T×Y 分层划分和匿名特征分组划分，并检查两种划分无泄漏。
    输入参数：
        - frame：已排除原 Test 的开发样本 DataFrame。
        - seed：本次开发划分随机种子。
        - outcome：用于分层和模型训练的二元结果列名。
    输出参数：
        - 字典：row_stratified 和 feature_group 两种划分各自的训练/验证位置索引二元组。
    """
    _validate_seeds((seed,))
    t, y = _binary_vectors(frame.treatment, frame[outcome])
    strata = t * 2 + y
    counts = np.unique(strata, return_counts=True)[1]
    if counts.min() < 2:
        raise ValueError(
            "each observed T x Y stratum needs at least two development rows"
        )
    positions = np.arange(len(frame))
    train, validation = train_test_split(
        positions, test_size=0.25, stratify=strata, random_state=seed
    )
    groups = feature_groups(frame)
    if len(np.unique(groups)) < 2:
        raise ValueError("feature-group sensitivity needs at least two feature groups")
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
        _binary_vectors(t[training], y[training])
        _binary_vectors(t[holdout], y[holdout])
        if np.intersect1d(training, holdout).size:
            raise AssertionError(f"{name} has overlapping row positions")
    if np.intersect1d(groups[group_train], groups[group_validation]).size:
        raise AssertionError("feature groups overlap across the group split")
    return result


def _counts(frame, outcome, prefix):
    """
    函数名：_counts
    功能：汇总指定 DataFrame 中 Control/Treatment 人数及对应结果正例数，并添加字段前缀。
    输入参数：
        - frame：需要统计的训练或验证 DataFrame。
        - outcome：统计的二元结果列名。
        - prefix：输出字段使用的前缀，例如 train 或 validation。
    输出参数：
        - 人数和正例数统计字典。
    """
    t, y = frame.treatment.to_numpy(), frame[outcome].to_numpy()
    return {
        f"{prefix}_n": len(frame),
        f"{prefix}_control_n": int((t == 0).sum()),
        f"{prefix}_treatment_n": int((t == 1).sum()),
        f"{prefix}_control_events": int(y[t == 0].sum()),
        f"{prefix}_treatment_events": int(y[t == 1].sum()),
    }


def _summarize(longmetrics):
    """
    函数名：_summarize
    功能：按划分方式、训练比例和策略汇总 Stability 指标的均值、标准差、极值及 Validation 选择频率。
    输入参数：
        - longmetrics：每次开发实验逐策略记录的长表 DataFrame。
    输出参数：
        - 二元组：(聚合指标 DataFrame、策略选择频率 DataFrame)。
    """
    dimensions = ["split_method", "training_fraction", "policy"]
    quantities = [
        "incremental_per_100k",
        "vs_random_per_100k",
        "vs_propensity_per_100k",
        "auuc_ht",
        "qini_ht",
        "qini_score_traditional",
        "fit_seconds",
        "train_n",
        "train_control_events",
        "validation_control_events",
    ]
    quantities += [
        name for name in ("auuc_plugin", "auuc_plugin_centered") if name in longmetrics
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
    cache,
    base_run,
    output,
    *,
    seeds=DEFAULT_SEEDS,
    fractions=DEFAULT_FRACTIONS,
    trees=150,
    primary_budget=0.2,
    outcome="conversion",
):
    """
    函数名：run_stability
    功能：在排除原 Test 的开发数据上运行多随机种子、训练规模和划分方式敏感性分析，并保存协议、明细和汇总。
    输入参数：
        - cache：当前样本 NPZ 缓存路径。
        - base_run：作为来源和旧 Test 行号依据的基准实验目录。
        - output：本次 Stability 新输出目录。
        - seeds：重复开发实验使用的随机种子序列。
        - fractions：行分层划分下使用的嵌套训练比例序列。
        - trees：Stability 中 XGBoost 模型的树数量。
        - primary_budget：比较策略时使用的主要触达预算。
        - outcome：模型训练和指标计算使用的结果列名。
    输出参数：
        - 完成后的 Stability 输出目录 pathlib.Path；同时写入 protocol、逐次指标、划分审计和汇总文件。
    """
    from .config import configure_runtime

    configure_runtime()
    seeds, fractions = _validate_seeds(seeds), _validate_fractions(fractions)
    if not isinstance(trees, (int, np.integer)) or isinstance(trees, bool) or trees < 1:
        raise ValueError("trees must be a positive integer")
    if not np.isfinite(primary_budget) or not 0 < primary_budget <= 1:
        raise ValueError("primary_budget must lie in (0, 1]")
    output = safe_output(output)
    base_run, cache = Path(base_run).resolve(), Path(cache).resolve()
    if (
        output == base_run
        or (base_run in output.parents and output != base_run / "stability")
        or output == cache.parent
        or cache.parent in output.parents
    ):
        raise ValueError(
            "output must be separate from the original run and sample cache"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output already exists and is nonempty: {output}")
    started = time.perf_counter()
    development, old_test_ids, provenance = load_development_sample(
        cache, base_run, outcome
    )
    # 在拟合任何模型前，先准备并校验全部划分。
    splits = {seed: development_splits(development, seed, outcome) for seed in seeds}
    protocol = {
        "study": "development_training_size_and_split_sensitivity_v1",
        "status": "frozen_before_model_fits",
        "seeds": seeds,
        "training_fractions": fractions,
        "primary_budget": primary_budget,
        "outcome": outcome,
        "features": FEATURES,
        "provenance": provenance,
        "candidates": POLICY_NAMES,
        "model_parameters": {
            candidate["name"]: make_uplift_model(
                **{k: v for k, v in candidate.items() if k != "name"},
                random_state=seeds[0],
                **({"n_estimators": trees} if candidate["outcome_model"] == "xgboost" else {}),
            ).get_params()
            for candidate in MODEL_CANDIDATES
        },
        "model_seed": "equal to the current split seed; all other model parameters are fixed",
        "row_split": "75/25 T x Y stratified development split; all configured nested fractions",
        "training_sampling": "within each T x Y stratum, one random permutation; floor(fraction*n_stratum), no replacement, no event oversampling",
        "group_split": "GroupShuffleSplit 75/25 distinct 64-bit hashes of 12 float32 features; full training fraction only; row counts may differ",
        "group_interpretation": "identical anonymous feature vectors are not established user identities; no automatic deduplication; hash collisions theoretically possible",
        "assignment_probability": "plug-in treatment fraction of the full training partition, shared across its nested fractions; validation outcomes do not estimate it",
        "primary_metric": "validation HT incremental conversions per 100000 eligible people at the realized floor(budget*n)/n fraction",
        "secondary_metrics": "full-resolution HT AUUC, centered HT Qini, and separately scaled traditional Qini score",
        "ranking_metric_definition_from_loaded_function": RANKING_METRIC_DESCRIPTION,
        "random_ranking": "analytic reference line: AUUC_HT=ATE_HT/2, centered Qini scores=0; not a sampled random ranking",
        "response_baseline": "propensity means P(Y=1|X,T=1), reused from t_xgboost; not P(T=1|X)",
        "selection": "one development winner per seed/split/fraction; stable ties follow candidate order; selection frequency is descriptive",
        "uncertainty": "mean/sample std/min/max over overlapping development runs; no confidence intervals, no bootstrap, no new independent test claim",
        "limitations": [
            "Protocol was specified after the original study, not before the original test was seen.",
            "Excluding old test rows prevents direct reuse but does not erase earlier analyst knowledge.",
            "The group holdout changes feature support; its values are sensitivity results, not a matched causal comparison of splitting methods.",
            "Rare-event counts and performance associations do not establish a causal explanation for model differences.",
            "Equal contact cost and conversion value are assumptions; public-sample assignment effects are not online revenue.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_result(protocol, output / "protocol.json")
    protocol_sha = fingerprint(output / "protocol.json")["sha256"]
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
                group_overlap = int(
                    np.intersect1d(feature_groups(train), val_groups).size
                )
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
                    "train_row_ids_sha256": row_ids_sha256(train.row_id),
                    "validation_row_ids_sha256": row_ids_sha256(validation.row_id),
                    "row_overlap": int(
                        np.intersect1d(train.row_id, validation.row_id).size
                    ),
                    "train_old_test_overlap": int(
                        np.isin(train.row_id, old_test_ids).sum()
                    ),
                    "validation_old_test_overlap": int(
                        np.isin(validation.row_id, old_test_ids).sum()
                    ),
                    "feature_group_overlap": group_overlap,
                    "validation_fraction_matching_train_features": float(
                        np.isin(val_groups, feature_groups(train)).mean()
                    ),
                }
                if any(
                    audit[name]
                    for name in (
                        "row_overlap",
                        "train_old_test_overlap",
                        "validation_old_test_overlap",
                    )
                ):
                    raise AssertionError("forbidden row leakage detected")
                if method == "feature_group" and group_overlap:
                    raise AssertionError("feature group leakage detected")
                audits.append(audit)
                score_dict, fit_seconds, ranks = {}, {}, {}
                fitted_models = {}
                for candidate in MODEL_CANDIDATES:
                    name = candidate["name"]
                    model = make_uplift_model(
                        **{k: v for k, v in candidate.items() if k != "name"},
                        random_state=seed,
                        **({"n_estimators": trees} if "xgboost" in name else {}),
                    )
                    fit_started = time.perf_counter()
                    model.fit(train[FEATURES], train.treatment, train[outcome])
                    fit_seconds[name] = time.perf_counter() - fit_started
                    fitted_models[name] = model
                score_dict = predict_policy_scores(fitted_models, validation[FEATURES])
                del fitted_models, model
                score_dict = {
                    name: score_dict[name] for name in POLICY_NAMES if name != "random"
                }
                for name, scores in score_dict.items():
                    ranks[name] = ranking_metrics(
                        validation[outcome], validation.treatment, scores, e, seed
                    )
                ate = ranks[MODEL_CANDIDATES[0]["name"]]["ate_ht"]
                ranks["random"] = {
                    "n": len(validation),
                    "ate_ht": ate,
                    "auuc_ht": ate / 2,
                    "qini_ht": 0.0,
                    "qini_score_traditional": 0.0,
                }
                policy_values = evaluate_policies(
                    validation[outcome],
                    validation.treatment,
                    score_dict,
                    e,
                    budgets=[primary_budget],
                    bootstrap=0,
                    seed=seed,
                )
                winner = (
                    policy_values.sort_values(
                        "incremental_per_100k", ascending=False, kind="stable"
                    )
                    .iloc[0]
                    .policy
                )
                for row in policy_values.to_dict("records"):
                    name = row["policy"]
                    records.append(
                        {
                            **{
                                k: audit[k]
                                for k in (
                                    "seed",
                                    "split_method",
                                    "training_fraction",
                                    "assignment_probability",
                                    "train_n",
                                    "train_control_n",
                                    "train_treatment_n",
                                    "train_control_events",
                                    "train_treatment_events",
                                    "validation_n",
                                    "validation_control_n",
                                    "validation_treatment_n",
                                    "validation_control_events",
                                    "validation_treatment_events",
                                )
                            },
                            **{
                                k: row[k]
                                for k in (
                                    "policy",
                                    "budget",
                                    "n_selected",
                                    "selected_fraction",
                                    "incremental_per_100k",
                                    "vs_random_per_100k",
                                    "vs_propensity_per_100k",
                                )
                            },
                            **ranks[name],
                            "fit_seconds": fit_seconds.get(name, 0.0),
                            "fit_source": (
                                "t_xgboost_reused" if name == "propensity" else name
                            ),
                            "selected_on_validation": name == winner,
                        }
                    )
                # 进度文件只保存在本次新目录中；预先冻结的分析配置保持不变。
                write_result(pd.DataFrame(records), output / "longmetrics.csv")
                write_result(pd.DataFrame(audits), output / "split_audit.csv")
    longmetrics = pd.DataFrame(records)
    aggregate, frequencies = _summarize(longmetrics)
    write_result(aggregate, output / "aggregate.csv")
    write_result(frequencies, output / "selection_frequency.csv")
    if fingerprint(output / "protocol.json")["sha256"] != protocol_sha:
        raise AssertionError("protocol changed after study started")
    # 结束前再次核验各只读输入文件的字节内容未改变。
    for label in ("cache", "cache_metadata", "base_manifest", "old_test_id_file"):
        if (
            fingerprint(provenance[label]["path"])["sha256"]
            != provenance[label]["sha256"]
        ):
            raise AssertionError(
                f"read-only input changed while study was running: {label}"
            )
    report = {
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "elapsed_seconds": time.perf_counter() - started,
        "experiments": len(audits),
        "fitted_models": len(audits) * len(MODEL_CANDIDATES),
        "development_n": len(development),
        "old_test_n_excluded": len(old_test_ids),
        "all_old_test_row_overlaps_zero": all(
            a["train_old_test_overlap"] == a["validation_old_test_overlap"] == 0
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
