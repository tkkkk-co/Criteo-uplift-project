"""Development-only training-size and split sensitivity study.

The old test set has already been inspected and is never used here. Repeated
development validations overlap; their standard deviations are descriptive,
not population confidence intervals or a new independent test result.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import FEATURES, fingerprint, load_sample, safe_output, write_json
from .evaluation import evaluate_policies, ranking_metrics
from .models import make_model

MODEL_NAMES = ("s_logistic", "t_logistic", "t_xgboost", "x_xgboost")
POLICY_NAMES = ("random", *MODEL_NAMES, "propensity")
DEFAULT_SEEDS = (17, 29, 83)
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)


def row_ids_sha256(ids):
    """Hash a row-ID set as sorted little-endian signed 64-bit integers."""
    return sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def _checked_ids(values, label):
    ids = np.asarray(values)
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu":
        raise ValueError(f"{label}: row IDs must be nonempty integer vectors")
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label}: row IDs must be unique and nonnegative")
    return ids.astype(np.int64)


def _binary_vectors(t, y):
    t, y = np.asarray(t), np.asarray(y)
    if t.ndim != 1 or y.shape != t.shape or not len(t):
        raise ValueError("t and y must be nonempty equal-length vectors")
    if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("t and y must contain only binary values")
    if len(np.unique(t)) != 2:
        raise ValueError("both treatment arms are required")
    return t.astype(np.int8), y.astype(np.int8)


def _validate_fractions(fractions):
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
    """64-bit hashes of twelve float32 model inputs, not inferred user IDs.

    Hash collisions are theoretically possible. No records are deduplicated;
    equal feature vectors are grouped only for the sensitivity split.
    """
    return pd.util.hash_pandas_object(
        frame[FEATURES].astype(np.float32), index=False
    ).to_numpy(dtype=np.uint64)


def load_development_sample(cache, base_run, outcome="conversion"):
    """Read the existing cache and subtract the entire previously exposed test.

    Verify cache/base provenance and ID membership. Reading metadata alone
    cannot prove the cache's contents against the original raw CSV; the cache
    byte fingerprint is therefore also frozen in the new protocol.
    """
    cache, base_run = Path(cache).resolve(), Path(base_run).resolve()
    if outcome not in ("conversion", "visit"):
        raise ValueError("outcome must be conversion or visit")
    manifest_path = base_run / "manifest.json"
    test_path = base_run / "test_row_ids.csv"
    metadata_path = cache.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if manifest.get("status") != "complete":
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
    if metadata.get("sample_size") not in (0, len(sample)):
        raise ValueError("cache row count disagrees with cache metadata")
    if not np.isfinite(sample[FEATURES].to_numpy()).all():
        raise ValueError("cached features must be finite for this study")
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
    """Nested, without-replacement proportional samples within T x Y strata.

    One independent random order per stratum is reused at all fractions.
    Each stratum contributes floor(fraction * stratum_size) records: no event
    oversampling or artificial minimum event count is imposed. As a result,
    selected size can be a few records below fraction * training size.
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
        # Sorting restores source order; membership is still random and nested.
        indices = np.sort(indices)
        if len(indices) == 0 or len(np.unique(t[indices])) != 2:
            raise ValueError(
                f"fraction {fraction} leaves too few rows or a missing treatment arm"
            )
        result[fraction] = indices
    return result


def development_splits(frame, seed=17, outcome="conversion"):
    """Return row-stratified 75/25 and feature-group 75/25 partitions.

    GroupShuffleSplit allocates 25% of distinct groups, so the validation
    *row* fraction need not be 25%. It is not outcome-stratified and targets
    a different holdout feature distribution from the row split.
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
    t, y = frame.treatment.to_numpy(), frame[outcome].to_numpy()
    return {
        f"{prefix}_n": len(frame),
        f"{prefix}_control_n": int((t == 0).sum()),
        f"{prefix}_treatment_n": int((t == 1).sum()),
        f"{prefix}_control_events": int(y[t == 0].sum()),
        f"{prefix}_treatment_events": int(y[t == 1].sum()),
    }


def _summarize(longmetrics):
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
    """Execute the frozen descriptive protocol and write a new artifact folder.

    Models are fitted only on development training rows and are not saved.
    Response propensity reuses T-XGBoost's treated outcome model. The full
    training partition supplies the fixed assignment probability for every
    nested fraction, keeping within-split HT comparisons on the same scale.
    """
    from .runtime import configure_runtime

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
        or base_run in output.parents
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
    # Prepare and validate all splits before starting any model fit.
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
            name: make_model(
                name,
                random_state=seeds[0],
                **({"n_estimators": trees} if "xgboost" in name else {}),
            ).get_params()
            for name in MODEL_NAMES
        },
        "model_seed": "equal to the current split seed; all other model parameters are fixed",
        "row_split": "75/25 T x Y stratified development split; all configured nested fractions",
        "training_sampling": "within each T x Y stratum, one random permutation; floor(fraction*n_stratum), no replacement, no event oversampling",
        "group_split": "GroupShuffleSplit 75/25 distinct 64-bit hashes of 12 float32 features; full training fraction only; row counts may differ",
        "group_interpretation": "identical anonymous feature vectors are not established user identities; no automatic deduplication; hash collisions theoretically possible",
        "assignment_probability": "plug-in treatment fraction of the full training partition, shared across its nested fractions; validation outcomes do not estimate it",
        "primary_metric": "validation HT incremental conversions per 100000 eligible people at the realized floor(budget*n)/n fraction",
        "secondary_metrics": "full-resolution HT AUUC, centered HT Qini, and separately scaled traditional Qini score",
        "ranking_metric_definition_from_loaded_function": ranking_metrics.__doc__,
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
    write_json(protocol, output / "protocol.json")
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
                for name in MODEL_NAMES:
                    model = make_model(
                        name,
                        random_state=seed,
                        **({"n_estimators": trees} if "xgboost" in name else {}),
                    )
                    fit_started = time.perf_counter()
                    model.fit(train[FEATURES], train.treatment, train[outcome])
                    fit_seconds[name] = time.perf_counter() - fit_started
                    score_dict[name] = model.predict(validation[FEATURES])
                    if name == "t_xgboost":
                        score_dict["propensity"] = model.predict_outcomes(
                            validation[FEATURES]
                        )[1]
                    del model
                # Reorder explicitly so tie handling does not depend on fit timing.
                score_dict = {
                    name: score_dict[name] for name in POLICY_NAMES if name != "random"
                }
                for name, scores in score_dict.items():
                    ranks[name] = ranking_metrics(
                        validation[outcome], validation.treatment, scores, e, seed
                    )
                ate = ranks[MODEL_NAMES[0]]["ate_ht"]
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
                # Progress files stay inside the new directory; protocol is immutable.
                pd.DataFrame(records).to_csv(output / "longmetrics.csv", index=False)
                pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False)
    longmetrics = pd.DataFrame(records)
    aggregate, frequencies = _summarize(longmetrics)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    frequencies.to_csv(output / "selection_frequency.csv", index=False)
    if fingerprint(output / "protocol.json")["sha256"] != protocol_sha:
        raise AssertionError("protocol changed after study started")
    # Verify each read-only input still has the same bytes before completion.
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
        "fitted_models": len(audits) * len(MODEL_NAMES),
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
    write_json(report, output / "report.json")
    return output
