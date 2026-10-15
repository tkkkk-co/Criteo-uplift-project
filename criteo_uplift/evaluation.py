"""Out-of-sample randomized-experiment and uplift-policy evaluation.

All scores must be produced by models fitted without the evaluated outcomes.
``e`` is a *fixed* randomized assignment probability (or its training-split
estimate), not a propensity refitted on the evaluation rows. HT values remain
distinct from the ordinary treated-minus-control difference in means.
"""

from __future__ import annotations

from math import erfc, sqrt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_BUDGETS = (0.02, 0.05, 0.10, 0.20, 0.50, 1.0)
_Z_975 = 1.959963984540054


def _binary_inputs(y, t):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.shape != t.shape or y.size == 0:
        raise ValueError(
            "y and t must be nonempty one-dimensional arrays of equal length"
        )
    if not np.isin(y, (0, 1)).all() or not np.isin(t, (0, 1)).all():
        raise ValueError("y and t must contain only finite binary values 0 and 1")
    if not np.any(t == 0) or not np.any(t == 1):
        raise ValueError("evaluation requires both treatment arms")
    return y, t.astype(np.int8)


def _scores(scores, n):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != n or not np.isfinite(values).all():
        raise ValueError("scores must be finite, one-dimensional, and match y")
    return values


def _ht_outcome(y, t, e):
    if not np.isscalar(e) or not np.isfinite(e) or not 0 < e < 1:
        raise ValueError(
            "e must be a fixed assignment probability strictly between 0 and 1"
        )
    return t * y / e - (1 - t) * y / (1 - e)


def _rank(scores, seed):
    # A separately seeded, outcome-independent tie break prevents CSV order
    # (which can group treatment arms) from defining a tied-score policy.
    tie_break = np.random.default_rng(seed).random(len(scores))
    return np.lexsort((tie_break, -scores))


def _mean_difference(y, t):
    treated = y[t == 1]
    control = y[t == 0]
    n1, n0 = len(treated), len(control)
    p1 = float(treated.mean()) if n1 else np.nan
    p0 = float(control.mean()) if n0 else np.nan
    ate = p1 - p0
    # Sample-variance / n: independent-arm standard error of a difference
    # in sample means. A singleton arm cannot estimate its variance.
    se = (
        sqrt(float(treated.var(ddof=1) / n1 + control.var(ddof=1) / n0))
        if min(n1, n0) > 1
        else np.nan
    )
    pooled_p = (float(treated.sum()) + float(control.sum())) / (n1 + n0)
    null_se = (
        sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n0)) if min(n1, n0) else np.nan
    )
    if null_se > 0:
        p_value = erfc(abs(ate / null_se) / sqrt(2))
    elif null_se == 0 and ate == 0:
        p_value = 1.0
    else:
        p_value = np.nan
    return {
        "n": n1 + n0,
        "n_treatment": n1,
        "n_control": n0,
        "response_treatment": p1,
        "response_control": p0,
        "ate": ate,
        "standard_error": se,
        "ci_low": ate - _Z_975 * se,
        "ci_high": ate + _Z_975 * se,
        "pooled_null_standard_error": null_se,
        "p_value": p_value,
        "relative_uplift": ate / p0 if p0 > 0 else np.nan,
    }


def ate_summary(y, t) -> dict:
    """Binary difference-in-means ATE, unpooled SE/Wald CI and pooled z test.

    Effects and relative uplift are fractions, not percentages. The two-sided
    null test pools conversion rates under H0. Relative uplift is undefined
    (NaN) when the control conversion rate is zero. A singleton arm yields
    NaN for the sample-variance SE and CI. Normal approximations can be poor
    for tiny event counts; no artificial clipping of the Wald CI is applied.
    """
    y, t = _binary_inputs(y, t)
    return _mean_difference(y, t)


def uplift_curve(y, t, scores, e: float, seed: int = 42) -> pd.DataFrame:
    """Return every ranked prefix, including the origin, of a frozen score.

    ``gain_ht[k] = sum(z[top k]) / n`` where
    ``z = t*y/e - (1-t)*y/(1-e)``. Its endpoint is the full-sample HT ATE.
    ``qini_traditional = (R1 - R0*N1/N0) / n`` is reported separately; it
    has a different scale and is NaN whenever the prefix has no controls.
    ``uplift`` is the diagnostic cumulative difference in arm means, and
    ``lift`` divides that diagnostic by the full-sample difference in means
    (NaN unless the full effect is positive and exceeds 1e-12). Both require both arms
    in the selected prefix. Neither replaces HT gain. The HT origin is zero.
    """
    y, t = _binary_inputs(y, t)
    n = len(y)
    scores = _scores(scores, n)
    z = _ht_outcome(y, t, e)
    order = _rank(scores, seed)
    sorted_t, sorted_y = t[order], y[order]
    count = np.arange(n + 1)
    n1 = np.r_[0, np.cumsum(sorted_t, dtype=np.int64)]
    n0 = count - n1
    r1 = np.r_[0.0, np.cumsum(sorted_t * sorted_y)]
    r0 = np.r_[0.0, np.cumsum((1 - sorted_t) * sorted_y)]
    traditional = np.full(n + 1, np.nan)
    has_control = n0 > 0
    traditional[has_control] = (
        r1[has_control] - r0[has_control] * n1[has_control] / n0[has_control]
    ) / n
    uplift = np.full(n + 1, np.nan)
    has_both = (n1 > 0) & (n0 > 0)
    uplift[has_both] = r1[has_both] / n1[has_both] - r0[has_both] / n0[has_both]
    lift = uplift / uplift[-1] if uplift[-1] > 1e-12 else np.full(n + 1, np.nan)
    plugin_gain = count / n * uplift
    plugin_gain[0] = 0.0
    gain = np.r_[0.0, np.cumsum(z[order])] / n
    result = pd.DataFrame(
        {
            "fraction": count / n,
            "n_selected": count,
            "gain_ht": gain,
            "random_gain_ht": count / n * gain[-1],
            "qini_traditional": traditional,
            "uplift": uplift,  # legacy name: cumulative arm-rate difference
            "cumulative_lift": uplift,
            "uplift_rate": uplift,
            "gain_plugin": plugin_gain,
            "lift": lift,  # legacy name: relative cumulative lift
            "relative_lift": lift,
            "n_treatment": n1,
            "n_control": n0,
            "responses_treatment": r1,
            "responses_control": r0,
        }
    )
    result.attrs.update({"assignment_probability": float(e), "tie_break_seed": seed})
    return result


def _area(x, values):
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x)))


def _ranking_from_ordered(y, t, e):
    """All rows already ordered, including repeated bootstrap observations."""
    n = len(y)
    cumulative = np.r_[0.0, np.cumsum(_ht_outcome(y, t, e))] / n
    auuc = float(np.sum((cumulative[:-1] + cumulative[1:]) * 0.5) / n)
    ate = float(cumulative[-1])
    count = np.arange(1, n + 1)
    n1 = np.cumsum(t, dtype=np.int64)
    n0 = count - n1
    r1 = np.cumsum(t * y)
    r0 = np.cumsum((1 - t) * y)
    # Q is algebraically zero when a prefix contains only controls. Keep
    # these defined nodes; previous v1 integrated only both-arm prefixes.
    control = n0 > 0
    q_fraction = np.r_[0.0, count[control] / n]
    traditional = np.r_[
        0.0, (r1[control] - r0[control] * n1[control] / n0[control]) / n
    ]
    both = (n1 > 0) & control
    rates = r1[both] / n1[both] - r0[both] / n0[both]
    fraction = np.r_[0.0, count[both] / n]
    plugin_gain = np.r_[0.0, count[both] / n * rates]
    plugin_area = _area(fraction, plugin_gain)
    return {
        "n": n,
        "auuc_ht": auuc,
        "qini_ht": auuc - ate / 2,
        "ate_ht": ate,
        "auuc_plugin": plugin_area,
        "auuc_plugin_centered": plugin_area - float(rates[-1]) / 2,
        "qini_score_traditional": _area(q_fraction, traditional)
        - float(traditional[-1]) / 2,
    }


def ranking_metrics(y, t, scores, e: float, seed: int = 42) -> dict:
    """Full-resolution areas, horizontal axis population fraction, v2.

    HT: G=sum(z[top k])/n, AUUC=integral G, qini_ht=AUUC-G(1)/2.
    Plug-in: L=R1/N1-R0/N0, G=(k/n)*L, AUUC=integral G.
    Traditional: Q=(R1-R0*N1/N0)/n, score=integral Q-Q(1)/2.
    Each area uses trapezoids at all defined prefixes and origin (0,0),
    bridging any initial undefined region linearly for integration only.
    Plug-in requires both arms; Q requires controls (Q=0 if N1=0).
    No perfect-ranking normalization. No downsampling of integration nodes.
    The v2 Q convention retains control-only zero nodes; historical v1
    artifacts used both-arm prefixes and are deliberately not overwritten.
    """
    y, t = _binary_inputs(y, t)
    scores = _scores(scores, len(y))
    order = _rank(scores, seed)
    return _ranking_from_ordered(y[order], t[order], e)


def ranking_confidence_intervals(
    y,
    t,
    score_dict: Mapping[str, Sequence[float]],
    e: float,
    bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired within-arm percentile intervals for frozen ranking scores.

    Reuse the same sampled rows for every model. Keep original independent
    tie priorities; duplicate sampled rows repeat at their original priority.
    Recompute whole curves, endpoints and centered areas in each replicate.
    Conditional on fitted models, e and observed arm sizes; does not include
    training/selection uncertainty, repeated users or multiplicity correction.
    Reference named 'propensity' is the response-score baseline. Zero draws
    returns point estimates and NaN intervals. Use >= 2 draws otherwise.
    """
    y, t = _binary_inputs(y, t)
    _ht_outcome(y, t, e)
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero or an integer of at least 2")
    if not score_dict or any(not isinstance(k, str) or not k for k in score_dict):
        raise ValueError("score_dict requires nonempty string policy names")
    names = list(score_dict)
    metrics = [
        "auuc_ht",
        "qini_ht",
        "auuc_plugin",
        "auuc_plugin_centered",
        "qini_score_traditional",
    ]
    n = len(y)
    orders = [_rank(_scores(score_dict[k], n), seed) for k in names]

    def values(order):
        result = _ranking_from_ordered(y[order], t[order], e)
        return [result[k] for k in metrics]

    point = np.asarray([values(order) for order in orders])
    boot = np.empty((bootstrap, len(names), len(metrics)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 271828]))
    arm1, arm0 = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    for b in range(bootstrap):
        draw1 = arm1[rng.integers(len(arm1), size=len(arm1))]
        draw0 = arm0[rng.integers(len(arm0), size=len(arm0))]
        counts = np.bincount(draw1, minlength=n) + np.bincount(draw0, minlength=n)
        for j, order in enumerate(orders):
            boot[b, j] = values(np.repeat(order, counts[order]))
    ci = (
        np.quantile(boot, [0.025, 0.975], axis=0)
        if bootstrap
        else np.full((2, *point.shape), np.nan)
    )
    ref = names.index("propensity") if "propensity" in names else None
    delta = point - point[ref] if ref is not None else np.full_like(point, np.nan)
    delta_ci = (
        np.quantile(boot - boot[:, ref : ref + 1], [0.025, 0.975], axis=0)
        if bootstrap and ref is not None
        else np.full_like(ci, np.nan)
    )
    return pd.DataFrame(
        [
            {
                "policy": name,
                "metric": metric,
                "estimate": point[j, m],
                "ci_low": ci[0, j, m],
                "ci_high": ci[1, j, m],
                "vs_propensity": delta[j, m],
                "vs_propensity_ci_low": delta_ci[0, j, m],
                "vs_propensity_ci_high": delta_ci[1, j, m],
                "bootstrap": bootstrap,
            }
            for j, name in enumerate(names)
            for m, metric in enumerate(metrics)
        ]
    )


def _prefix_values(z_sorted, counts_sorted, k):
    """Weighted top-k sums; bootstrap multiplicities reselect the boundary."""
    cumulative_n = np.cumsum(counts_sorted)
    cumulative_z = np.cumsum(counts_sorted * z_sorted)
    selected = np.zeros(len(k), dtype=float)
    positive = k > 0
    boundary = np.searchsorted(cumulative_n, k[positive], side="left")
    before_n = np.where(boundary > 0, cumulative_n[np.maximum(boundary - 1, 0)], 0)
    before_z = np.where(boundary > 0, cumulative_z[np.maximum(boundary - 1, 0)], 0.0)
    selected[positive] = before_z + (k[positive] - before_n) * z_sorted[boundary]
    return selected


def evaluate_policies(
    y,
    t,
    score_dict: Mapping[str, Sequence[float]],
    e: float,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare top-k policies to random and a response-score baseline.

    Name the response-score baseline ``propensity`` for paired differences to
    it (this name refers to *response* propensity, never treatment assignment).
    The reserved ``random`` policy is added automatically. Budget q permits
    k=floor(q*n) contacts; its realized fraction k/n is used for all policies.
    Random is an analytic expected policy value, (k/n)*ATE_HT, with no extra
    Monte Carlo assignment noise. Values are incremental responses per 100k
    eligible people, not per 100k contacted people; costs are not modeled.

    Each bootstrap sample resamples individuals independently *within arm*,
    holding arm sizes and e fixed. The same sample is used for every policy
    and for random, so difference intervals are paired percentile intervals.
    Learned scores and independent tie priorities are frozen. Every replicate
    reselects its top-k rows; multinomial multiplicities plus sorted-prefix
    sums implement that resampling without repeated sorting. Uncertainty is
    conditional on trained models and excludes their training variability.
    Bootstrap=0 returns estimates with NaN CIs; otherwise use at least 2 draws.
    """
    y, t = _binary_inputs(y, t)
    n = len(y)
    z = _ht_outcome(y, t, e)
    budget = np.asarray(budgets, dtype=float)
    if (
        budget.ndim != 1
        or not len(budget)
        or not np.isfinite(budget).all()
        or np.any((budget < 0) | (budget > 1))
    ):
        raise ValueError(
            "budgets must be a nonempty sequence of fractions between 0 and 1"
        )
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0 or bootstrap == 1:
        raise ValueError("bootstrap must be zero or an integer of at least 2")
    if "random" in score_dict:
        raise ValueError("'random' is reserved for the analytic random-policy baseline")
    if any(not isinstance(name, str) or not name for name in score_dict):
        raise ValueError("policy names must be nonempty strings")
    k = np.floor(budget * n + 1e-12).astype(np.int64)
    fractions = k / n
    names = ["random", *score_dict]
    orders = [_rank(_scores(score_dict[name], n), seed) for name in names[1:]]
    sorted_z = [z[order] for order in orders]
    point = np.empty((len(names), len(budget)))
    point[0] = fractions * z.mean()
    for j, values in enumerate(sorted_z, start=1):
        cumulative = np.r_[0.0, np.cumsum(values)]
        point[j] = cumulative[k] / n
    # All policies coincide at full coverage. Use the shared aggregate to
    # prevent accumulation order from creating spurious floating-point gains.
    point[:, k == n] = point[0, k == n]

    boot = np.empty((bootstrap, len(names), len(budget)))
    if bootstrap:
        # Independent of the RNG used for tie priorities.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
        arm1 = np.flatnonzero(t == 1)
        arm0 = np.flatnonzero(t == 0)
        for b in range(bootstrap):
            draws1 = arm1[rng.integers(len(arm1), size=len(arm1))]
            draws0 = arm0[rng.integers(len(arm0), size=len(arm0))]
            counts = np.bincount(draws1, minlength=n) + np.bincount(draws0, minlength=n)
            boot[b, 0] = fractions * np.dot(counts, z) / n
            for j, (order, values) in enumerate(zip(orders, sorted_z), start=1):
                boot[b, j] = _prefix_values(values, counts[order], k) / n
            boot[b, :, k == n] = boot[b, 0, k == n, None]

    scale = 100_000.0
    point *= scale
    boot *= scale
    nan_intervals = np.full((2, len(names), len(budget)), np.nan)
    ci = np.quantile(boot, [0.025, 0.975], axis=0) if bootstrap else nan_intervals
    random_delta = point - point[0]
    random_ci = (
        np.quantile(boot - boot[:, :1], [0.025, 0.975], axis=0)
        if bootstrap
        else nan_intervals
    )
    response_index = names.index("propensity") if "propensity" in names else None
    propensity_delta = (
        point - point[response_index]
        if response_index is not None
        else np.full_like(point, np.nan)
    )
    propensity_ci = (
        np.quantile(
            boot - boot[:, response_index : response_index + 1], [0.025, 0.975], axis=0
        )
        if bootstrap and response_index is not None
        else nan_intervals
    )
    records = []
    for j, name in enumerate(names):
        for b, requested in enumerate(budget):
            records.append(
                {
                    "policy": name,
                    "budget": float(requested),
                    "n_selected": int(k[b]),
                    "selected_fraction": float(fractions[b]),
                    "incremental_per_100k": float(point[j, b]),
                    "ci_low": float(ci[0, j, b]),
                    "ci_high": float(ci[1, j, b]),
                    "vs_random_per_100k": float(random_delta[j, b]),
                    "vs_random_ci_low": float(random_ci[0, j, b]),
                    "vs_random_ci_high": float(random_ci[1, j, b]),
                    "vs_propensity_per_100k": float(propensity_delta[j, b]),
                    "vs_propensity_ci_low": float(propensity_ci[0, j, b]),
                    "vs_propensity_ci_high": float(propensity_ci[1, j, b]),
                    "bootstrap_replicates": int(bootstrap),
                }
            )
    result = pd.DataFrame.from_records(records)
    result.attrs.update(
        {
            "assignment_probability": float(e),
            "ci_method": "paired arm-stratified percentile bootstrap; fixed learned scores; top-k reselected",
            "random_policy": "analytic expected value at the realized selected fraction",
            "seed": seed,
        }
    )
    return result


def calibration_table(y, t, cate, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count prediction bins, with independent-arm observed ATE/CI.

    Bins increase from low to high predicted CATE and use only predictions to
    define their membership. Ties are broken independently using seed 42;
    identical predictions may span bins. A bin with a missing arm has an
    undefined observed ATE. Bin CIs are descriptive pointwise Wald intervals,
    not simultaneous bands. Do not use these evaluation outcomes to refit or
    recalibrate the reported model.
    """
    y, t = _binary_inputs(y, t)
    cate = _scores(cate, len(y))
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    order = _rank(-cate, seed=42)
    records = []
    for number, selected in enumerate(
        np.array_split(order, min(n_bins, len(y))), start=1
    ):
        summary = _mean_difference(y[selected], t[selected])
        records.append(
            {
                "bin": number,
                "n": len(selected),
                "n_treatment": summary["n_treatment"],
                "n_control": summary["n_control"],
                "predicted_cate": float(cate[selected].mean()),
                "prediction_min": float(cate[selected].min()),
                "prediction_max": float(cate[selected].max()),
                "observed_ate": summary["ate"],
                "standard_error": summary["standard_error"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
            }
        )
    return pd.DataFrame.from_records(records)
