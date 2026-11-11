
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .config import FEATURES
from .input_checks import check_feature_matrix, check_binary_vector
from .pipeline_io import safe_output, write_result

_PRETREATMENT_FEATURES = frozenset(FEATURES)

def _features(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    #-------------
    # 函数名：_features
    # 功能：校验特征列名并按指定顺序提取处理前特征矩阵。
    # 输入参数：
    #     - df：包含待检查特征列的 DataFrame。
    #     - features：要提取的处理前特征列名序列。
    # 输出参数：
    #     - 特征矩阵：按 features 顺序排列的 float64 NumPy 二维数组。
    #-------------
    names = list(features)
    if not names or len(names) != len(set(names)):
        raise ValueError("features 必须为非空且无重复项的序列")
    invalid = set(names) - _PRETREATMENT_FEATURES
    if invalid:
        raise ValueError(f"仅允许使用前置特征 f0--f11，发现非法列：{sorted(invalid)}")
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"缺少特征列：{sorted(missing)}")
    values = df.loc[:, names].to_numpy(dtype=np.float64, copy=True)
    check_feature_matrix(values)
    return values



def srm_summary(
    n1: int,
    n0: int,
    expected_ratio: float = 0.85,
) -> dict:
    #-------------
    # 函数名：srm_summary
    # 功能：将处理组和对照组人数与指定的处理组参考比例进行 SRM 卡方检验。
    # 输入参数：
    #     - n1：Treatment 组实际人数。
    #     - n0：Control 组实际人数。
    #     - expected_ratio：预期 Treatment 人数占总人数的比例，默认 0.85。
    # 输出参数：
    #     - SRM 摘要字典：包含组人数、实际/参考比例、卡方统计量、p 值和解释。
    #-------------
    if any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0 # 布尔值属于整数子类，需要显式排除。
        for n in (n1, n0)
    ):
        raise ValueError("处理组和对照组人数必须为非负整数")

    n = int(n1 + n0)
    if n == 0:
        raise ValueError("至少需要一条观测记录")

    if (
        expected_ratio is None
        or not np.isfinite(expected_ratio)
        or not 0 < expected_ratio < 1
    ):
        raise ValueError(
            "expected_ratio 必须为严格介于 0 和 1 之间的有限参考比例"
        )

    expected1, expected0 = (n * expected_ratio,n * (1 - expected_ratio))
    test = chisquare(
        f_obs=[n1,n0],
        f_exp=[expected1,expected0],
    )
    return {
        "n_treatment": int(n1),
        "n_control": int(n0),
        "observed_ratio": float(n1 / n),
        "expected_ratio": float(expected_ratio),
        "chi2": float(test.statistic),
        "pvalue": float(test.pvalue),
        "status": "computed",
        "reason": (
            "Pearson chi-square count check against the supplied reference ratio; "
            "this alone does not validate original experiment randomization."
            + (
                " Expected arm count below 5; asymptotic p-value may be inaccurate."
                if min(expected1, expected0) < 5
                else ""
            )
        ),
    }


def covariate_balance(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    #-------------
    # 函数名：covariate_balance
    # 功能：逐个特征计算处理组与对照组的均值、SMD 和双样本 KS 统计量。
    # 输入参数：
    #     - df：包含 treatment 和待检查特征的 DataFrame。
    #     - features：需要检查的处理前特征列名序列。
    # 输出参数：
    #     - 平衡诊断表：每个特征一行，包含两组统计量、SMD 和 KS 结果。
    #-------------
    x = _features(df, features)
    check_binary_vector(df["treatment"], "treatment", len(x))
    treatment = df["treatment"].to_numpy(dtype=np.int8)
    n1, n0 = int(treatment.sum()), int((treatment == 0).sum())
    if min(n1, n0) < 2:
        raise ValueError("协变量平衡诊断要求每个处理臂至少有两行数据")
    records = []

    for j, feature in enumerate(features):
        treated, control = x[treatment == 1, j], x[treatment == 0, j]
        mean1, mean0 = float(treated.mean()), float(control.mean())
        scale = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        # 计算标准化均值差（SMD）。
        if scale > 0:
            smd = (mean1 - mean0) / scale
        else:
            smd = 0.0 if mean1 == mean0 else np.copysign(np.inf, mean1 - mean0)
        # 计算两样本 KS 统计量及 p 值。
        ks = ks_2samp(treated, control, alternative="two-sided", method="asymp")

        records.append(
            {
                "feature": feature,
                "n_treatment": n1,
                "n_control": n0,
                "mean_treatment": mean1,
                "mean_control": mean0,
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    return pd.DataFrame.from_records(records)


def _make_c2st_model(name: str, seed: int):
    #-------------
    # 函数名：_make_c2st_model
    # 功能：根据模型名称创建 C2ST 使用的 Logistic 或 XGBoost 分类器。
    # 输入参数：
    #     - name：分类器类型，支持 logistic 或 xgboost。
    #     - seed：模型随机种子。
    # 输出参数：
    #     - 未拟合的 scikit-learn 兼容分类器。
    #-------------
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, random_state=seed),
        )
    if name == "xgboost":
        # 沿用 30 轮、最多 15 叶的复杂度预算；学习率 0.1、L2=1。
        # 单线程避免大量置换拟合时过度订阅；配置预先固定，不根据留出 AUC 或 P 值调参。
        # min_child_weight 是 Hessian 权重阈值，不等同于 HGB 的最少叶样本数。
        from .config import configure_runtime
        configure_runtime()
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=30,
            max_depth=0,
            max_leaves=15,
            grow_policy="lossguide",
            min_child_weight=1.0,
            learning_rate=0.1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"不支持的模型：{name}")


def _probability_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    #-------------
    # 函数名：_probability_metrics
    # 功能：根据真实二元标签和正类预测概率计算概率分类评价指标。
    # 输入参数：
    #     - y：真实的 0/1 标签向量。
    #     - prediction：每行预测为正类的概率向量。
    # 输出参数：
    #     - 指标字典：包含 ROC AUC、Average Precision、LogLoss 和 Brier score。
    #-------------
    # y：真实的二分类标签。
    # prediction：每一行预测为 1 的概率。

    both_classes = np.unique(y).size == 2
    return {
        "roc_auc": float(roc_auc_score(y, prediction)) if both_classes else np.nan,
        "average_precision": (
            float(average_precision_score(y, prediction)) if np.any(y == 1) else np.nan
        ),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prediction)),
    }


def c2st(
    df: pd.DataFrame, # 主流程通常传入已抽好的 20,000 行，T/C 为 1:1。
    features: Sequence[str],
    sample_size: int = 20_000,
    permutations: int = 99,
    seed: int = 42,
) -> dict:
    #-------------
    # 函数名：c2st
    # 功能：执行处理组/对照组可区分性检验，并通过重复置换标签、完整重拟合模型构造 AUC 零分布。
    # 输入参数：
    #     - df：包含 treatment 和特征列的候选诊断样本。
    #     - features：用于区分处理组的处理前特征列名。
    #     - sample_size：本次 C2ST 最多使用的样本行数。
    #     - permutations：标签置换并重新拟合模型的次数。
    #     - seed：抽样、数据划分和模型的随机种子。
    # 输出参数：
    #     - 结果字典：包含观测 summary、置换 null_auc、样本规模、随机种子及解释说明。
    #-------------
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, (int, np.integer))
        or sample_size < 4
    ):
        raise ValueError("sample_size 必须为大于等于 4 的整数")
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, (int, np.integer))
        or permutations < 1
    ):
        raise ValueError("permutations 必须为正整数")


    rng = np.random.default_rng(seed)
    positions = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sample = df.iloc[positions]
    x = _features(sample, features)
    check_binary_vector(sample["treatment"], "treatment", len(x))
    t = sample["treatment"].to_numpy(dtype=np.int8)

    if len(t) < 4 or np.min(np.bincount(t, minlength=2)) < 2:
        raise ValueError("C2ST 的训练集和测试集都必须包含处理组与对照组")

    train_idx, test_idx = train_test_split(
        np.arange(len(t)), test_size=0.3, stratify=t, random_state=seed
    )
    x_train, x_test = x[train_idx], x[test_idx]
    t_train, t_test = t[train_idx], t[test_idx]
    if min(np.unique(t_train).size, np.unique(t_test).size) != 2:
        raise ValueError("C2ST 划分后仅有一个处理臂，请增加样本量")

    # 常数基线：每个留出样本都预测训练集的 Treatment 占比；本次等量样本中为 0.5。
    dummy = _probability_metrics(t_test, np.full(len(t_test), t_train.mean()))

    # 用真实标签分别训练 Logistic 和 XGBoost，记录留出集的观测指标。
    observed = {}
    for name in ("logistic", "xgboost"):
        classifier = _make_c2st_model(name, seed)
        classifier.fit(x_train, t_train)
        observed[name] = _probability_metrics(
            t_test, classifier.predict_proba(x_test)[:, 1]
        )
    # 在训练集与留出集内分别打乱标签，每轮重新拟合，构造置换零分布。
    null_records = []
    for b in range(permutations):
        shuffled_train = rng.permutation(t_train)
        shuffled_test = rng.permutation(t_test)
        for name in ("logistic", "xgboost"):
            classifier = _make_c2st_model(name, seed)
            classifier.fit(x_train, shuffled_train)
            null_records.append(
                {
                    "model": name,
                    "permutation": b + 1,
                    "roc_auc": float(
                        roc_auc_score(
                            shuffled_test, classifier.predict_proba(x_test)[:, 1]
                        )
                    ),
                }
            )
    # null_auc：将每个模型、每次置换对应的 AUC 整理为结果表。
    null_auc = pd.DataFrame.from_records(null_records)

    records = []
    for name, metrics in observed.items():
        null = null_auc.loc[null_auc["model"] == name, "roc_auc"].to_numpy()
        # 上尾检验：统计零分布中 AUC 至少与观测值一样高的次数，计算置换 p 值。
        pvalue = (1 + np.count_nonzero(null >= metrics["roc_auc"])) / (permutations + 1)
        records.append(
            {
                "model": name,
                **metrics,
                "dummy_log_loss": dummy["log_loss"],
                "dummy_roc_auc": dummy["roc_auc"],
                "dummy_brier": dummy["brier"],
                "train_treatment_ratio": float(t_train.mean()),
                "test_treatment_ratio": float(t_test.mean()),
                "pvalue": float(pvalue),
                "null_auc_mean": float(null.mean()),
                "permutations": int(permutations),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary["pvalue_holm"] = multipletests(summary["pvalue"], method="holm")[1]
    return {
        "summary": summary,
        "null_auc": null_auc,
        "sample_n": len(t),
        "train_n": len(t_train),
        "test_n": len(t_test),
        "seed": seed,
        "interpretation": (
            "Exploratory check of observable assignment predictability on development data. "
            "A nonsignificant result does not prove randomization. A significant result "
            "requires investigation of design, sampling, and feature timing. "
            "Permutation exchangeability assumes an unstratified random assignment mechanism; "
            "unknown blocking or multiple experiments may require a different permutation scheme."
        ),
    }


def learnability(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    outcome: str,
    by_arm: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    #-------------
    # 函数名：learnability
    # 功能：使用训练集拟合 Dummy、Logistic 和 XGBoost 结果模型，并在验证集评估整体或分处理臂的结果可预测性。
    # 输入参数：
    #     - train：用于拟合结果模型的训练集。
    #     - val：用于评价结果模型的验证集。
    #     - features：模型使用的处理前特征列名。
    #     - outcome：要预测的二元结果列名，支持 conversion 或 visit。
    #     - by_arm：是否额外按 Treatment 和 Control 分别拟合、评价。
    #     - seed：Logistic/XGBoost 模型随机种子。
    # 输出参数：
    #     - Learnability 结果表：按分析范围和模型记录样本量及概率预测指标。
    #-------------
    if outcome not in {"conversion", "visit"}:
        raise ValueError("outcome 必须为 conversion 或 visit")
    x_train, x_val = _features(train, features), _features(val, features)
    for frame in (train, val):
        for column in (outcome, "treatment"):
            check_binary_vector(frame[column], column, len(frame))
    y_train, y_val = train[outcome].to_numpy(dtype=np.int8), val[outcome].to_numpy(dtype=np.int8)
    t_train, t_val = train.treatment.to_numpy(dtype=np.int8), val.treatment.to_numpy(dtype=np.int8)
    scopes = [("whole", np.ones(len(train), dtype=bool), np.ones(len(val), dtype=bool))]
    if by_arm:
        scopes.extend(
            [
                ("treatment", t_train == 1, t_val == 1),
                ("control", t_train == 0, t_val == 0),
            ]
        )
    records = []
    for scope, train_mask, val_mask in scopes:
        xt, xv = x_train[train_mask], x_val[val_mask]
        yt, yv = y_train[train_mask], y_val[val_mask]
        metadata = {
            "scope": scope,
            "outcome": outcome,
            "n_train": len(yt),
            "n_validation": len(yv),
            "positive_rate_train": float(yt.mean()) if len(yt) else np.nan,
            "positive_rate_validation": float(yv.mean()) if len(yv) else np.nan,
        }
        for name in ("dummy", "logistic", "xgboost"):
            record = {**metadata, "model": name}
            if len(yt) == 0 or len(yv) == 0:
                records.append({**record, "status": "skipped_empty_scope"})
                continue
            if name == "dummy":
                prediction = np.full(len(yv), yt.mean())
            elif np.unique(yt).size < 2:
                records.append({**record, "status": "skipped_single_class_train"})
                continue
            else:
                classifier = _make_c2st_model(name, seed)
                classifier.fit(xt, yt)
                prediction = classifier.predict_proba(xv)[:, 1]
            records.append(
                {**record, "status": "computed", **_probability_metrics(yv, prediction)}
            )
    return pd.DataFrame.from_records(records)


def run_diagnostics(
    out, *, audit, source, scanned, train, validation, sample_size=20_000, permutations=499,
    seed=42, progress=print,
):
    #-------------
    # 函数名：run_diagnostics
    # 功能：组织全量数据的 SRM、协变量平衡、C2ST 置换检验和结果可学习性诊断，并保存诊断产物。
    # 输入参数：
    #     - out：本次实验的 diagnostics 输出目录。
    #     - audit：Preparation 生成的全量审计字典。
    #     - source：本次原始数据文件的指纹信息。
    #     - scanned：流式扫描生成的平衡统计、C2ST 样本和 metadata。
    #     - train：建模样本的训练集。
    #     - validation：建模样本的验证集。
    #     - sample_size：C2ST 诊断样本总行数。
    #     - permutations：C2ST 完整重拟合置换次数。
    #     - seed：诊断抽样和模型随机种子。
    #     - progress：接收进度文本的回调函数。
    # 输出参数：
    #     - 诊断详情字典；同时在 out 目录写入 SRM、平衡、C2ST 和 Learnability 结果。
    #-------------
    if sample_size < 20 or sample_size % 20:
        raise ValueError("C2ST sample_size 必须大于等于 20 且能被 20 整除")
    if permutations < 1:
        raise ValueError("permutations 必须为正数")
    out = safe_output(out)
    if not out.is_dir():
        raise FileNotFoundError(out)
    started = time.perf_counter()
    balance, sample, metadata = scanned['balance'], scanned['c2st_sample'], scanned['metadata']
    if metadata is None or balance is None or sample is None or scanned['srm'] is None:
        raise ValueError('扫描结果未包含完整诊断数据')
    if metadata['seed'] != seed or metadata['sample_n'] != sample_size:
        raise ValueError('扫描结果的 C2ST 样本规模或种子与当前参数不同')
    if metadata["source"]["sha256"] != source["sha256"]:
        raise ValueError("诊断与当前实验的原始数据版本不同")
    for key in ("n_treatment", "n_control"):
        if metadata[key] != audit[key]:
            raise ValueError(f"全量诊断计数与 audit 不一致：{key}")
    from threadpoolctl import threadpool_limits

    progress(f"全量来源 C2ST：两组各 {sample_size // 2:,} 行，{permutations} 次完整重拟合置换")
    # 将本次诊断使用的线程数限制为 4，控制完整重拟合置换的计算资源。
    with threadpool_limits(limits=4):
        check = c2st(sample, FEATURES, sample_size, permutations, seed)
    summary = check["summary"].assign(
        scope="full_source_arm_balanced_sample", representation="raw_csv_float64",
        metric="ROC AUC", sample_n=sample_size,
        n_control=sample_size // 2, n_treatment=sample_size // 2,
        pvalue_resolution=1 / (permutations + 1),
        source_sha256=metadata["source"]["sha256"],
    )
    if not (summary[["train_treatment_ratio", "test_treatment_ratio"]] == .5).all().all():
        raise AssertionError("C2ST 内部两组人数不平衡")
    details = {
        **{key: value for key, value in check.items() if not isinstance(value, pd.DataFrame)},
        "scope": "full_source_arm_balanced_sample", "source": metadata["source"],
        "representation": "raw_csv_float64", "features": FEATURES,
        "full_n": metadata["n"], "full_n_control": metadata["n_control"],
        "full_n_treatment": metadata["n_treatment"], "rows_dropped": metadata["rows_dropped"],
        "sampling": metadata["sampling"], "sampling_target": metadata["sampling_target"],
        "n_control": sample_size // 2, "n_treatment": sample_size // 2,
        "train_n_control": check["train_n"] // 2, "train_n_treatment": check["train_n"] // 2,
        "holdout_n_control": check["test_n"] // 2, "holdout_n_treatment": check["test_n"] // 2,
        "metric": "one-sided ROC AUC", "multiplicity": "Holm across two prespecified classifiers",
        "models": ["logistic", "xgboost"],
        "xgboost_parameters": _make_c2st_model("xgboost", seed).get_params(),
        "permutations": permutations, "pvalue_resolution": 1 / (permutations + 1),
        "sample_row_ids_sha256": hashlib.sha256(sample.row_id.to_numpy().tobytes()).hexdigest(),
        "ks_method": metadata["ks_method"], "elapsed_seconds": time.perf_counter() - started,
        "interpretation": (
            "全量原始数据组内等概率、两组等量抽样，比较P(X|T=1)与P(X|T=0)。"
            "不显著不证明联合分布相同或随机化；显著不单独证明随机化失败。"
            "置换依赖可交换性，未知分块、多实验和实体依赖未验证。"
            "这是事后描述性诊断，不用于正式模型选择。"
        ),
    }
    # 所有计算及来源核对完成后，直接写入调用方传入的诊断目录。
    write_result(balance, out / 'balance.csv')
    write_result(summary, out / 'c2st.csv')
    write_result(check["null_auc"].assign(scope=details["scope"]), out / 'c2st_null.csv')
    write_result(details, (out / 'c2st_details.json'))
    write_result(scanned['srm'], (out / 'srm.json'))
    records = [learnability(train, validation, FEATURES, outcome, seed=seed)
               for outcome in ("visit", "conversion")]
    write_result(pd.concat(records), out / 'learnability.csv')
    return details
