"""Read-only diagnostics for the Criteo randomized uplift dataset.

These checks use only pre-treatment f0--f11. Predicting assignment diagnoses
observable imbalance; it cannot establish randomization or rule out hidden bias.
Outcome learnability is not a test for treatment-effect heterogeneity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_distribution
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
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

_PRETREATMENT_FEATURES = frozenset(f"f{i}" for i in range(12))


def _features(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    names = list(features)
    if not names or len(names) != len(set(names)):
        raise ValueError("features must be a nonempty sequence without duplicates")
    invalid = set(names) - _PRETREATMENT_FEATURES
    if invalid:
        raise ValueError(f"Only pre-treatment f0--f11 are allowed: {sorted(invalid)}")
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    values = df.loc[:, names].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite; handle missing values explicitly")
    return values


def _binary(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        raise ValueError(f"Missing column: {column}")
    values = df[column].to_numpy(copy=True)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{column} must contain only binary 0/1 values")
    return values.astype(np.int8)


def _model(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, random_state=seed),
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=30,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            learning_rate=0.1,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def _probability_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    # Supplying both labels keeps log loss meaningful for a single-class holdout.
    both_classes = np.unique(y).size == 2
    return {
        "roc_auc": float(roc_auc_score(y, prediction)) if both_classes else np.nan,
        "average_precision": (
            float(average_precision_score(y, prediction)) if np.any(y == 1) else np.nan
        ),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prediction)),
    }


def _holm(pvalues: Sequence[float]) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    ordered = pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))
    result = np.empty_like(pvalues)
    result[order] = np.minimum(1.0, np.maximum.accumulate(ordered))
    return result


def covariate_balance(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Compare treatment arms using pooled-SD SMD and two-sided KS tests.

    Intended for the training/development sample. SMD is signed treatment minus
    control, with denominator sqrt((s1**2 + s0**2)/2). KS p-values are unadjusted
    exploratory diagnostics, not a pass/fail proof of randomization.
    """
    x = _features(df, features)
    treatment = _binary(df, "treatment")
    n1, n0 = int(treatment.sum()), int((treatment == 0).sum())
    if min(n1, n0) < 2:
        raise ValueError("Covariate balance needs at least two rows in each arm")
    records = []
    for j, feature in enumerate(features):
        treated, control = x[treatment == 1, j], x[treatment == 0, j]
        mean1, mean0 = float(treated.mean()), float(control.mean())
        scale = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        if scale > 0:
            smd = (mean1 - mean0) / scale
        else:
            smd = 0.0 if mean1 == mean0 else np.copysign(np.inf, mean1 - mean0)
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


def srm_summary(n1: int, n0: int, expected_ratio: float | None = None) -> dict:
    """Check allocation counts only against a supplied design treatment share.

    expected_ratio means P(T=1), not the treatment/control count ratio. The
    observed allocation must never be substituted as its own null hypothesis.
    """
    if any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0
        for n in (n1, n0)
    ):
        raise ValueError("Arm counts must be nonnegative integers")
    n = int(n1 + n0)
    if n == 0:
        raise ValueError("At least one observation is required")
    result = {
        "n_treatment": int(n1),
        "n_control": int(n0),
        "observed_ratio": float(n1 / n),
        "expected_ratio": expected_ratio,
        "chi2": None,
        "pvalue": None,
    }
    if expected_ratio is None:
        return {
            **result,
            "status": "skipped",
            "reason": "Design allocation is unknown; the observed ratio cannot be its own SRM null.",
        }
    if not np.isfinite(expected_ratio) or not 0 < expected_ratio < 1:
        raise ValueError(
            "expected_ratio must be a design probability strictly between 0 and 1"
        )
    expected1, expected0 = n * expected_ratio, n * (1 - expected_ratio)
    statistic = (n1 - expected1) ** 2 / expected1 + (n0 - expected0) ** 2 / expected0
    return {
        **result,
        "expected_ratio": float(expected_ratio),
        "chi2": float(statistic),
        "pvalue": float(chi2_distribution.sf(statistic, df=1)),
        "status": "computed",
        "reason": (
            "Pearson chi-square allocation check against the supplied design ratio."
            + (
                " Expected arm count below 5; asymptotic p-value may be inaccurate."
                if min(expected1, expected0) < 5
                else ""
            )
        ),
    }


def c2st(
    df: pd.DataFrame,
    features: Sequence[str],
    sample_size: int = 20_000,
    permutations: int = 99,
    seed: int = 42,
) -> dict:
    """Out-of-sample assignment prediction with a complete refit permutation null.

    Uniform sampling uses row positions, independently of T and Y. Observations
    are then fixed in a stratified 70/30 train/test split. For each permutation,
    assignment labels are independently shuffled within these two sets, keeping
    each arm count fixed, and BOTH classifiers are refit from scratch. The test
    is one-sided in ROC AUC (no adaptive score flipping). Holm adjusts across
    the two prespecified classifiers. No hyperparameter selection is performed.
    """
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, (int, np.integer))
        or sample_size < 4
    ):
        raise ValueError("sample_size must be an integer of at least 4")
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, (int, np.integer))
        or permutations < 1
    ):
        raise ValueError("permutations must be a positive integer")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sample = df.iloc[positions]
    x = _features(sample, features)
    t = _binary(sample, "treatment")
    if len(t) < 4 or np.min(np.bincount(t, minlength=2)) < 2:
        raise ValueError("C2ST needs both treatment arms represented in train and test")
    train_idx, test_idx = train_test_split(
        np.arange(len(t)), test_size=0.3, stratify=t, random_state=seed
    )
    x_train, x_test = x[train_idx], x[test_idx]
    t_train, t_test = t[train_idx], t[test_idx]
    if min(np.unique(t_train).size, np.unique(t_test).size) != 2:
        raise ValueError("C2ST split has a single arm; supply a larger sample")
    dummy = _probability_metrics(t_test, np.full(len(t_test), t_train.mean()))
    observed = {}
    for name in ("logistic", "hgb"):
        classifier = _model(name, seed)
        classifier.fit(x_train, t_train)
        observed[name] = _probability_metrics(
            t_test, classifier.predict_proba(x_test)[:, 1]
        )
    null_records = []
    for b in range(permutations):
        shuffled_train = rng.permutation(t_train)
        shuffled_test = rng.permutation(t_test)
        for name in ("logistic", "hgb"):
            classifier = _model(name, seed)
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
    null_auc = pd.DataFrame.from_records(null_records)
    records = []
    for name, metrics in observed.items():
        null = null_auc.loc[null_auc["model"] == name, "roc_auc"].to_numpy()
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
    summary["pvalue_holm"] = _holm(summary["pvalue"])
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
    """Predict a binary outcome using X, with fixed logistic/HGB and dummy models.

    The dummy predicts the TRAIN outcome rate. Preprocessing is fit on training
    rows only. Optional arm-specific models use separate arm training/validation
    rows. Low predictive skill is not evidence of absent effect heterogeneity.
    Single-class training sets report skipped fitted models and retain dummy
    evaluation; ROC AUC is undefined for a single-class validation set.
    """
    if outcome not in {"conversion", "visit"}:
        raise ValueError("outcome must be conversion or visit")
    x_train, x_val = _features(train, features), _features(val, features)
    y_train, y_val = _binary(train, outcome), _binary(val, outcome)
    t_train, t_val = _binary(train, "treatment"), _binary(val, "treatment")
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
        for name in ("dummy", "logistic", "hgb"):
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
                classifier = _model(name, seed)
                classifier.fit(xt, yt)
                prediction = classifier.predict_proba(xv)[:, 1]
            records.append(
                {**record, "status": "computed", **_probability_metrics(yv, prediction)}
            )
    return pd.DataFrame.from_records(records)
