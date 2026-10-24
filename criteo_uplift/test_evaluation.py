"""Evaluate frozen uplift models on the held-out test split.

This module consumes already-fitted models and does not train models or change
the policy selected on the validation split.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, write_json
from .evaluation import (
    ate_summary,
    calibration_table,
    evaluate_policies,
    ranking_confidence_intervals,
    ranking_metrics,
    uplift_curve,
)


# ----------------------------------------------------------------------------
# Function: evaluate_test_set — evaluate frozen candidates on the test split.
# Inputs:
#     test: held-out DataFrame with features, treatment, and the chosen outcome.
#     models: fitted candidate models trained without test rows.
#     outcome: primary binary outcome column.
#     assignment_probability: fixed treatment-assignment probability estimate.
#     seed, bootstrap: reproducibility seed and bootstrap replicate count.
#     primary_budget: budget selected for the primary policy comparison.
#     out: directory for test metrics, curves, calibration, and predictions.
#     progress: callback accepting one progress-message string.
# Outputs:
#     DataFrame of test policy metrics; also writes the established test artifacts.
# Notes:
#     Test metrics are for reporting only; this function never changes selection.
# ----------------------------------------------------------------------------
def evaluate_test_set(
    test,
    models,
    *,
    outcome,
    assignment_probability,
    seed,
    bootstrap,
    primary_budget,
    out,
    progress=print,
):
    """Evaluate already-fitted models without reselecting the primary policy.

    Args:
        test: held-out test DataFrame.
        models: mapping from candidate name to fitted model.
        outcome: binary outcome column.
        assignment_probability: fixed assignment-probability estimate.
        seed: deterministic seed for ranking and bootstrap procedures.
        bootstrap: number of paired bootstrap replicates.
        primary_budget: primary contact fraction chosen on validation data.
        out: directory for test artifacts.
        progress: callback used for progress messages.

    Returns:
        DataFrame of policy metrics saved as test_policies.csv.
    """
    out = Path(out)
    test_scores = {
        name: model.predict(test[FEATURES])
        for name, model in models.items()
    }
    test_scores["propensity"] = models["t_xgboost"].predict_outcomes(
        test[FEATURES]
    )[1]

    rankings = []
    curves = []
    for name, score in test_scores.items():
        rankings.append(
            {
                "policy": name,
                **ranking_metrics(
                    test[outcome],
                    test.treatment,
                    score,
                    assignment_probability,
                    seed,
                ),
                "mean_score": float(np.mean(score)),
                "negative_fraction": float(np.mean(score < 0)),
            }
        )
        curve = uplift_curve(
            test[outcome],
            test.treatment,
            score,
            assignment_probability,
            seed,
        )
        # Keep full-resolution metrics; thin only the saved plotting table.
        curve = curve.iloc[
            np.unique(np.linspace(0, len(curve) - 1, 501).astype(int))
        ].copy()
        curve.insert(0, "policy", name)
        curves.append(curve)
        if name != "propensity":
            calibration_table(test[outcome], test.treatment, score).to_csv(
                out / f"calibration_{name}.csv", index=False
            )

    pd.DataFrame(rankings).to_csv(out / "test_ranking.csv", index=False)
    pd.concat(curves).to_csv(out / "curves.csv", index=False)

    budgets = sorted(set([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, primary_budget]))
    policies = evaluate_policies(
        test[outcome],
        test.treatment,
        test_scores,
        assignment_probability,
        budgets=budgets,
        bootstrap=bootstrap,
        seed=seed,
    )
    policies.to_csv(out / "test_policies.csv", index=False)
    progress("计算冻结模型的配对 AUUC / Qini 区间")
    ranking_confidence_intervals(
        test[outcome],
        test.treatment,
        test_scores,
        assignment_probability,
        bootstrap=bootstrap,
        seed=seed,
    ).to_csv(out / "ranking_intervals.csv", index=False)

    write_json(
        ate_summary(test[outcome], test.treatment),
        out / "ate_test.json",
    )
    np.savez_compressed(
        out / "predictions.npz",
        row_id=test.row_id.to_numpy(),
        y=test[outcome].to_numpy(),
        t=test.treatment.to_numpy(),
        **test_scores,
    )
    return policies
