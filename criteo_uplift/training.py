"""Fit candidate uplift models and select a policy using validation data.

The module deliberately does not inspect test outcomes. Model selection is
frozen before test-set evaluation begins.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .data import FEATURES, write_json
from .diagnostics import learnability
from .evaluation import evaluate_policies, ranking_metrics
from .models import make_model


# ----------------------------------------------------------------------------
# Function: run_learnability_diagnostics — assess outcome predictability.
# Inputs:
#     train, validation: split DataFrames with FEATURES and the outcome labels.
#     out: directory where learnability.csv will be written.
#     seed: deterministic model seed.
# Outputs:
#     None; writes learnability.csv.
# Notes:
#     This evaluates prediction of observed outcomes, not treatment-effect
#     heterogeneity or CATE quality.
# ----------------------------------------------------------------------------
def run_learnability_diagnostics(train, validation, out, *, seed):
    """Run the existing outcome-learnability check for visit and conversion.

    Args:
        train: training split.
        validation: validation split.
        out: output directory for the CSV artifact.
        seed: random seed forwarded to the diagnostic models.

    Returns:
        None. Writes learnability.csv.
    """
    records = [
        learnability(train, validation, FEATURES, outcome, seed=seed)
        for outcome in ("visit", "conversion")
    ]
    pd.concat(records).to_csv(Path(out) / "learnability.csv", index=False)


# ----------------------------------------------------------------------------
# Function: fit_and_select_models — fit candidates and freeze policy selection.
# Inputs:
#     train, validation: mutually exclusive modeling data splits.
#     outcome: primary binary outcome used for fitting and ranking.
#     seed, trees: model randomness and XGBoost tree count.
#     primary_budget: contact fraction used for the primary validation decision.
#     assignment_probability: fixed treatment-assignment probability estimate.
#     out: directory for model, validation, and selection artifacts.
#     progress: callback accepting a progress-message string.
# Outputs:
#     dict with fitted models, model metadata, and the frozen selection record.
# Notes:
#     Models are fitted on train only; validation selects the policy; test is
#     not passed to this function and therefore cannot affect selection.
# ----------------------------------------------------------------------------
def fit_and_select_models(
    train,
    validation,
    *,
    outcome,
    seed,
    trees,
    primary_budget,
    assignment_probability,
    out,
    progress=print,
):
    """Fit the four uplift candidates and freeze the validation-selected policy.

    Args:
        train, validation: disjoint training and validation DataFrames.
        outcome: binary outcome column used by the run.
        seed: deterministic seed for model fitting and ranking metrics.
        trees: number of estimators for XGBoost candidates.
        primary_budget: contact fraction used for the primary decision.
        assignment_probability: training-split estimate of P(T=1).
        out: output directory for model and validation artifacts.
        progress: callback used to report model-training progress.

    Returns:
        Dictionary with keys models, model_metadata, and selection.
    """
    out = Path(out)
    models = {}
    validation_scores = {}
    validation_ranking = []
    model_metadata = {}
    model_names = ["s_logistic", "t_logistic", "t_xgboost", "x_xgboost"]

    for name in model_names:
        progress("训练 " + name)
        kwargs = {"n_estimators": trees} if "xgboost" in name else {}
        model = make_model(name, random_state=seed, **kwargs)
        started = time.perf_counter()
        model.fit(train[FEATURES], train.treatment, train[outcome])
        model_metadata[name] = {
            "parameters": model.get_params(),
            "seconds": time.perf_counter() - started,
        }
        validation_scores[name] = model.predict(validation[FEATURES])
        models[name] = model
        validation_ranking.append(
            {
                "policy": name,
                **ranking_metrics(
                    validation[outcome],
                    validation.treatment,
                    validation_scores[name],
                    assignment_probability,
                    seed,
                ),
            }
        )
        joblib.dump(model, out / f"{name}.joblib")

    # Response propensity is P(Y=1 | X,T=1), not P(T=1 | X).
    validation_scores["propensity"] = models["t_xgboost"].predict_outcomes(
        validation[FEATURES]
    )[1]
    validation_policies = evaluate_policies(
        validation[outcome],
        validation.treatment,
        validation_scores,
        assignment_probability,
        budgets=[primary_budget],
        bootstrap=0,
        seed=seed,
    )
    validation_policies.to_csv(out / "validation_policies.csv", index=False)
    pd.DataFrame(validation_ranking).to_csv(
        out / "validation_ranking.csv", index=False
    )

    # Freeze both the best available policy and the best uplift-only candidate.
    selected_policy = (
        validation_policies.sort_values(
            "incremental_per_100k", ascending=False, kind="stable"
        )
        .iloc[0]
        .policy
    )
    selected_uplift_model = (
        validation_policies[validation_policies.policy.isin(model_names)]
        .sort_values("incremental_per_100k", ascending=False, kind="stable")
        .iloc[0]
        .policy
    )
    selection = {
        "primary_budget": primary_budget,
        "criterion": "validation HT incremental responses per 100k eligible people",
        "selected_policy": selected_policy,
        "selected_uplift_model": selected_uplift_model,
        "candidates": ["random", "propensity", *model_names],
        "frozen_before_test": True,
        "note": (
            "All models remain trained on training split only; no test tuning, "
            "no train+validation refit."
        ),
    }
    write_json(selection, out / "selection.json")
    return {
        "models": models,
        "model_metadata": model_metadata,
        "selection": selection,
    }
