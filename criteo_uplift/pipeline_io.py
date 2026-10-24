"""Write pipeline artifacts and finalize a completed or diagnostics-only run.

The functions in this module persist results using the existing artifact names
consumed by Notebooks 01–06. They do not fit models.
"""

from pathlib import Path
import importlib.metadata
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

from .data import FEATURES, ROOT, safe_output, write_json
from .diagnostics import update_experiment_diagnostics
from .evaluation import aggregate_ate_from_audit
from .reporting import build_report


# ----------------------------------------------------------------------------
# Function: write_split_artifacts — persist split summaries and original row IDs.
# Inputs:
#     parts: mapping of train / validation / test to their DataFrames.
#     out: output directory for this experiment.
# Outputs:
#     None; writes the six-row splits.csv summary and one row-ID CSV per split.
# ----------------------------------------------------------------------------
def write_split_artifacts(parts, out):
    """Save split-level counts and original row identifiers.

    Args:
        parts: mapping with train, validation, and test DataFrames.
        out: output directory.

    Returns:
        None. Writes splits.csv and *_row_ids.csv files.
    """
    out = Path(out)
    split_summary = []
    for name, frame in parts.items():
        frame[["row_id"]].to_csv(out / f"{name}_row_ids.csv", index=False)
        for arm in (0, 1):
            rows = frame[frame.treatment == arm]
            split_summary.append(
                {
                    "split": name,
                    "treatment": arm,
                    "n": len(rows),
                    "conversion_positive": int(rows.conversion.sum()),
                    "visit_positive": int(rows.visit.sum()),
                }
            )
    pd.DataFrame(split_summary).to_csv(out / "splits.csv", index=False)


# ----------------------------------------------------------------------------
# Function: write_feature_overlap — measure exact feature-vector matches to train.
# Inputs:
#     parts: mapping of train / validation / test DataFrames.
#     out: output directory for feature_overlap.json.
# Outputs:
#     Dictionary of validation/test matching fractions; also writes JSON.
# Notes:
#     Uses only FEATURES (f0–f11), not row_id, treatment, or outcome labels.
# ----------------------------------------------------------------------------
def write_feature_overlap(parts, out):
    """Write the fraction of validation/test feature vectors found in train.

    Args:
        parts: mapping with train, validation, and test DataFrames.
        out: output directory.

    Returns:
        Mapping with validation and test matching fractions.
    """
    train = parts["train"]
    validation = parts["validation"]
    test = parts["test"]
    train_hash = pd.util.hash_pandas_object(
        train[FEATURES], index=False
    ).to_numpy()
    overlap = {
        name: float(
            np.isin(
                pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy(),
                train_hash,
            ).mean()
        )
        for name, frame in (("validation", validation), ("test", test))
    }
    write_json(
        {
            "note": (
                "Identical anonymized feature vectors, not identified users; "
                "based on modeling float32 representation"
            ),
            "fraction_matching_training_features": overlap,
        },
        Path(out) / "feature_overlap.json",
    )
    return overlap


# ----------------------------------------------------------------------------
# Function: write_audit_artifacts — save the full-source audit and ATE summary.
# Inputs:
#     audit: full-source aggregate audit returned by scan_raw_data().
#     out: output directory for audit.json and ate_full.csv.
# Outputs:
#     None; writes audit.json and one aggregate ATE row per outcome.
# ----------------------------------------------------------------------------
def write_audit_artifacts(audit, out):
    """Persist the full-source audit and aggregate ATE estimates.

    Args:
        audit: audit dictionary computed from the full source.
        out: output directory.

    Returns:
        None. Writes audit.json and ate_full.csv.
    """
    out = Path(out)
    write_json(audit, out / "audit.json")
    pd.DataFrame(
        [aggregate_ate_from_audit(audit, outcome) for outcome in ("conversion", "visit")]
    ).to_csv(out / "ate_full.csv", index=False)


# ----------------------------------------------------------------------------
# Function: write_run_manifest — persist run parameters, assumptions, and versions.
# Inputs:
#     out: output directory.
#     args: parsed CLI arguments for this run.
#     assignment_probability: training-split estimate of P(T=1).
#     selection: frozen validation policy selection.
#     model_metadata: model parameters and fit durations.
#     source: raw-source fingerprint.
#     sample_n: number of rows across train / validation / test.
#     started_at: perf_counter value recorded at run start.
# Outputs:
#     None; writes manifest.json with status metrics_complete.
# ----------------------------------------------------------------------------
def write_run_manifest(
    out,
    *,
    args,
    assignment_probability,
    selection,
    model_metadata,
    source,
    sample_n,
    started_at,
):
    """Write the reproducibility manifest before report generation.

    Args:
        out: output directory.
        args: parsed run configuration.
        assignment_probability: fixed assignment-probability estimate.
        selection: frozen policy-selection record.
        model_metadata: fitted model parameters and elapsed fit times.
        source: source-file fingerprint metadata.
        sample_n: total number of modeling rows across all splits.
        started_at: monotonic run-start timestamp.

    Returns:
        None. Writes manifest.json with status metrics_complete.
    """
    write_json(
        {
            "assignment_probability": assignment_probability,
            "probability_source": "training treatment fraction; plug-in estimate",
            "ci_method": (
                "paired within-arm percentile bootstrap, fixed learned scores and "
                "assignment probability; top-k reselected"
            ),
            "uncertainty_excluded": [
                "model training and selection",
                "assignment probability estimation",
                "unidentified user dependence",
            ],
            "estimand": (
                "public-sample assignment (ITT-style) incremental outcome relative "
                "to treat-none"
            ),
            "cost_assumption": "equal cost/value; budget is contact fraction, not currency",
            "ranking_metric_version": (
                "v2; Qini retains control-only zero prefixes; plug-in AUUC explicit; "
                "no perfect normalization"
            ),
            "primary_outcome": args.outcome,
            "selection": selection,
            "models": model_metadata,
            "assumptions": [
                "released sample has approximately constant assignment probability conditional on X",
                "independent rows, consistency and no interference",
                "equal value per conversion and equal contact cost",
            ],
            "status": "metrics_complete",
            "source": source,
            "features": FEATURES,
            "seed": args.seed,
            "sample_n": sample_n,
            "arguments": vars(args),
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "xgboost",
                    "matplotlib",
                    "joblib",
                )
            },
            "elapsed_seconds": time.perf_counter() - started_at,
        },
        Path(out) / "manifest.json",
    )


# ----------------------------------------------------------------------------
# Function: finalize_run — build the report and mark the run complete.
# Inputs:
#     out: output directory with all metric artifacts.
#     policies: test-set policy metrics DataFrame.
#     primary_budget: frozen primary contact fraction.
#     progress: callback accepting one progress-message string.
# Outputs:
#     out: completed run directory.
# Notes:
#     Report generation only reads saved outputs; it does not refit models.
# ----------------------------------------------------------------------------
def finalize_run(out, policies, primary_budget, progress=print):
    """Generate the HTML report, update manifest status, and show primary results.

    Args:
        out: output directory.
        policies: test policy metrics.
        primary_budget: primary contact fraction.
        progress: progress-message callback.

    Returns:
        Path to the completed run directory.
    """
    out = Path(out)
    from .reporting import build_report

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


# ----------------------------------------------------------------------------
# Function: update_diagnostics_only — refresh diagnostics in an existing run.
# Inputs:
#     args: CLI configuration with data, seed, C2ST sample, and permutations.
#     root: project root used to resolve the default run directory.
#     progress: callback accepting one progress-message string.
# Outputs:
#     out: existing run directory after diagnostic files/report are refreshed.
# Notes:
#     Does not split data, refit uplift models, or rewrite model-selection results.
# ----------------------------------------------------------------------------
def update_diagnostics_only(args, *, root=ROOT, progress=print):
    """Refresh diagnostics in place without retraining or changing model selection.

    Args:
        args: parsed CLI configuration.
        root: project root used for the default output path.
        progress: progress-message callback.

    Returns:
        Path to the updated run directory.
    """
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
