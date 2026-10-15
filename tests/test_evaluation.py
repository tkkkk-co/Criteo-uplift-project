"""Small hand-checkable cases for the causal estimators and paired bootstrap."""

import math
import unittest

import numpy as np

from criteo_uplift.evaluation import (
    _prefix_values,
    ate_summary,
    calibration_table,
    evaluate_policies,
    ranking_metrics,
    ranking_confidence_intervals,
    uplift_curve,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        self.t = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        self.scores = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)

    def test_difference_in_means_and_pooled_null(self):
        result = ate_summary(self.y, self.t)
        self.assertEqual(result["ate"], 0.5)
        self.assertEqual(result["relative_uplift"], 2.0)
        self.assertAlmostEqual(result["standard_error"], math.sqrt(0.125))
        self.assertAlmostEqual(result["p_value"], math.erfc(1.0))
        self.assertLess(result["ci_low"], result["ate"])
        self.assertGreater(result["ci_high"], result["ate"])

    def test_constant_outcomes_and_missing_control_response(self):
        result = ate_summary(np.zeros(8), self.t)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["standard_error"], 0.0)
        self.assertTrue(np.isnan(result["relative_uplift"]))

    def test_ht_curve_preserves_fixed_assignment_probability(self):
        curve = uplift_curve(self.y, self.t, self.scores, e=0.8)
        expected_z = self.t * self.y / 0.8 - (1 - self.t) * self.y / 0.2
        np.testing.assert_allclose(curve.gain_ht, np.r_[0.0, expected_z.cumsum()] / 8)
        self.assertNotAlmostEqual(
            curve.gain_ht.iloc[-1], ate_summary(self.y, self.t)["ate"]
        )
        self.assertTrue(np.isnan(curve.qini_traditional.iloc[2]))
        self.assertAlmostEqual(curve.qini_traditional.iloc[3], -1 / 8)
        self.assertAlmostEqual(curve.uplift.iloc[3], -0.5)
        self.assertAlmostEqual(curve.lift.iloc[-1], 1)
        self.assertTrue(np.isnan(curve.uplift.iloc[2]))

    def test_area_uses_full_curve_and_centering(self):
        curve = uplift_curve(self.y, self.t, self.scores, e=0.5)
        metrics = ranking_metrics(self.y, self.t, self.scores, e=0.5)
        expected = sum(
            (curve.gain_ht.iloc[i] + curve.gain_ht.iloc[i + 1]) / 16 for i in range(8)
        )
        self.assertAlmostEqual(metrics["auuc_ht"], expected)
        self.assertAlmostEqual(metrics["qini_ht"], expected - metrics["ate_ht"] / 2)

    def test_traditional_qini_bridges_initial_undefined_prefix(self):
        y, t = [1, 0, 0, 1], [1, 1, 0, 0]
        scores = [4, 3, 2, 1]
        curve = uplift_curve(y, t, scores, e=0.5)
        metrics = ranking_metrics(y, t, scores, e=0.5)
        # Valid Qtrad nodes are (3/4, 1/4), (1, 0). With origin the
        # trapezoids have areas 3/32 and 1/32, and the random line is zero.
        self.assertTrue(curve.qini_traditional.iloc[:3].isna().all())
        self.assertEqual(curve.qini_traditional.iloc[-1], 0)
        self.assertEqual(metrics["qini_score_traditional"], 1 / 8)

    def test_traditional_qini_centers_its_own_full_coverage_endpoint(self):
        y, t = [1, 0, 0, 0], [1, 1, 0, 0]
        scores = [4, 3, 2, 1]
        curve = uplift_curve(y, t, scores, e=0.6)
        metrics = ranking_metrics(y, t, scores, e=0.6)
        # Qtrad ends at 1/4. Raw area = 3/32 + 1/16 = 5/32;
        # subtract random triangular area 1/8 to obtain 1/32.
        self.assertEqual(curve.qini_traditional.iloc[-1], 1 / 4)
        self.assertEqual(metrics["qini_score_traditional"], 1 / 32)
        self.assertNotEqual(metrics["qini_score_traditional"], metrics["qini_ht"])

    def test_traditional_score_is_not_changed_by_ht_assignment_probability(self):
        a = ranking_metrics(self.y, self.t, self.scores, e=0.5)
        b = ranking_metrics(self.y, self.t, self.scores, e=0.8)
        self.assertEqual(a["qini_score_traditional"], b["qini_score_traditional"])

    def test_tie_break_is_seeded_and_does_not_use_row_order(self):
        n = 100
        t = np.r_[np.ones(n // 2), np.zeros(n // 2)]
        y = np.ones(n)
        a = uplift_curve(y, t, np.ones(n), e=0.5, seed=42)
        b = uplift_curve(y, t, np.ones(n), e=0.5, seed=42)
        c = uplift_curve(y, t, np.ones(n), e=0.5, seed=43)
        np.testing.assert_array_equal(a.gain_ht, b.gain_ht)
        self.assertFalse(np.array_equal(a.gain_ht, c.gain_ht))
        self.assertGreater(a.n_control.iloc[10], 0)

    def test_weighted_prefix_reselects_boundary(self):
        # Expanded sorted sample is [3, 3, -2, -2, -2, 7].
        z = np.array([10.0, 3.0, -2.0, 7.0])
        counts = np.array([0, 2, 3, 1])
        k = np.array([0, 1, 2, 3, 5, 6])
        expected = np.r_[0, np.cumsum(np.repeat(z, counts))][k]
        np.testing.assert_array_equal(_prefix_values(z, counts, k), expected)

    def test_same_score_paired_difference_is_exactly_zero(self):
        result = evaluate_policies(
            self.y,
            self.t,
            {"propensity": self.scores, "T": self.scores},
            e=0.5,
            budgets=[0.25, 0.5, 1.0],
            bootstrap=40,
            seed=19,
        )
        same = result[result.policy == "T"]
        np.testing.assert_array_equal(same.vs_propensity_per_100k, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_low, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_high, 0)
        full = result[result.budget == 1]
        np.testing.assert_allclose(full.incremental_per_100k, 50_000)
        np.testing.assert_allclose(full.vs_random_per_100k, 0, atol=1e-10)
        np.testing.assert_allclose(full.vs_random_ci_low, 0, atol=1e-10)
        np.testing.assert_allclose(full.vs_random_ci_high, 0, atol=1e-10)

    def test_bootstrap_is_reproducible_and_point_estimates_unchanged(self):
        kwargs = dict(
            y=self.y,
            t=self.t,
            score_dict={"S": self.scores},
            e=0.5,
            budgets=[0.5],
            seed=77,
        )
        a = evaluate_policies(**kwargs, bootstrap=30)
        b = evaluate_policies(**kwargs, bootstrap=30)
        point = evaluate_policies(**kwargs, bootstrap=0)
        np.testing.assert_array_equal(a.ci_low, b.ci_low)
        np.testing.assert_array_equal(a.ci_high, b.ci_high)
        np.testing.assert_array_equal(
            a.incremental_per_100k, point.incremental_per_100k
        )
        self.assertTrue(point.ci_low.isna().all())
        self.assertTrue(point.vs_propensity_per_100k.isna().all())

    def test_budget_rounding_and_analytic_random(self):
        result = evaluate_policies(
            self.y, self.t, {"S": self.scores}, e=0.5, budgets=[0.01, 0.3], bootstrap=0
        )
        random = result[result.policy == "random"]
        np.testing.assert_array_equal(random.n_selected, [0, 2])
        np.testing.assert_array_equal(random.selected_fraction, [0, 0.25])
        np.testing.assert_array_equal(random.incremental_per_100k, [0, 12_500])

    def test_paired_difference_matches_direct_resampling(self):
        # Independent explicit row expansion checks optimized bootstrap math.
        budgets = np.array([0.25, 0.5])
        seed, replicates = 15, 30
        result = evaluate_policies(
            self.y,
            self.t,
            {"S": self.scores},
            e=0.5,
            budgets=budgets,
            bootstrap=replicates,
            seed=seed,
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
        z = 2 * self.t * self.y - 2 * (1 - self.t) * self.y
        arm1, arm0 = np.flatnonzero(self.t == 1), np.flatnonzero(self.t == 0)
        delta = []
        for _ in range(replicates):
            sample = np.r_[
                arm1[rng.integers(len(arm1), size=len(arm1))],
                arm0[rng.integers(len(arm0), size=len(arm0))],
            ]
            sample = sample[np.argsort(-self.scores[sample], kind="stable")]
            gain = np.r_[0, np.cumsum(z[sample])][(budgets * 8).astype(int)] / 8
            delta.append((gain - budgets * z[sample].mean()) * 100_000)
        expected = np.quantile(delta, [0.025, 0.975], axis=0)
        selected = result[result.policy == "S"]
        np.testing.assert_allclose(selected.vs_random_ci_low, expected[0])
        np.testing.assert_allclose(selected.vs_random_ci_high, expected[1])

    def test_calibration_bins_cover_data_and_missing_arms_are_nan(self):
        bins = calibration_table(self.y, self.t, self.scores, n_bins=4)
        self.assertEqual(bins.n.sum(), len(self.y))
        self.assertTrue(bins.predicted_cate.is_monotonic_increasing)
        self.assertTrue(bins.observed_ate.isna().all())
        single = calibration_table(self.y, self.t, self.scores, n_bins=1)
        self.assertEqual(single.observed_ate.iloc[0], 0.5)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            ate_summary([1, 0], [1, 1])
        with self.assertRaises(ValueError):
            ate_summary([1, np.nan], [1, 0])
        with self.assertRaises(ValueError):
            uplift_curve(self.y, self.t, self.scores, e=1)
        with self.assertRaises(ValueError):
            ranking_metrics(self.y, self.t, [1, 2], e=0.5)
        with self.assertRaises(ValueError):
            evaluate_policies(
                self.y, self.t, {"random": self.scores}, e=0.5, bootstrap=0
            )
        with self.assertRaises(ValueError):
            evaluate_policies(self.y, self.t, {}, e=0.5, budgets=[1.2], bootstrap=0)


class ExtendedMetricTests(unittest.TestCase):
    def setUp(self):
        self.t = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        self.y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        self.s = np.arange(8, 0, -1)

    def test_full_eight_row_hand_example(self):
        c = uplift_curve(self.y, self.t, self.s, 0.5)
        np.testing.assert_allclose(
            c.cumulative_lift.iloc[2:], [1, 1, 1, 2 / 3, 1 / 3, 5 / 12, 1 / 2]
        )
        np.testing.assert_allclose(
            c.gain_plugin.iloc[2:], [1 / 4, 3 / 8, 1 / 2, 5 / 12, 1 / 4, 35 / 96, 1 / 2]
        )
        np.testing.assert_allclose(
            c.qini_traditional.iloc[2:],
            [1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8, 5 / 24, 1 / 4],
        )
        m = ranking_metrics(self.y, self.t, self.s, 0.5)
        for name, expected in {
            'auuc_plugin': 81 / 256,
            'auuc_plugin_centered': 17 / 256,
            'qini_score_traditional': 19 / 384,
            'auuc_ht': 3 / 8,
            'qini_ht': 1 / 8,
        }.items():
            self.assertAlmostEqual(m[name], expected)

    def test_qini_keeps_defined_control_only_zero_nodes(self):
        # Q nodes: (0,0),(.25,0),(.5,.25),(.75,.25),(1,.5).
        # Area 3/16 minus random area 1/4 = -1/16.
        m = ranking_metrics([0, 1, 0, 1], [0, 1, 0, 1], [4, 3, 2, 1], 0.5)
        self.assertEqual(m['qini_score_traditional'], -1 / 16)

    def test_relative_lift_undefined_for_nonpositive_total_effect(self):
        for y in [np.zeros(8), 1 - self.y]:
            c = uplift_curve(y, self.t, self.s, 0.5)
            self.assertTrue(c.relative_lift.isna().all())
            self.assertTrue(np.isfinite(c.gain_plugin.iloc[-1]))

    def test_paired_ranking_intervals_same_score_zero_difference(self):
        r = ranking_confidence_intervals(
            self.y, self.t, {'propensity': self.s, 'same': self.s}, 0.5, 30
        )
        same = r[r.policy == 'same']
        np.testing.assert_array_equal(same.vs_propensity, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_low, 0)
        np.testing.assert_array_equal(same.vs_propensity_ci_high, 0)

    def test_ranking_bootstrap_matches_independent_expanded_sample(self):
        seed, draws = 19, 40
        score_dict = {'propensity': self.s, 'reverse': -self.s}
        r = ranking_confidence_intervals(self.y, self.t, score_dict, 0.5, draws, seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 271828]))
        a1, a0 = np.flatnonzero(self.t == 1), np.flatnonzero(self.t == 0)
        expected = []
        for _ in range(draws):
            ix = np.r_[rng.choice(a1, len(a1)), rng.choice(a0, len(a0))]
            # Repeated identical rows have equal outcomes so tie order is irrelevant.
            rev = ranking_metrics(self.y[ix], self.t[ix], -self.s[ix], 0.5)[
                'qini_score_traditional'
            ]
            ref = ranking_metrics(self.y[ix], self.t[ix], self.s[ix], 0.5)[
                'qini_score_traditional'
            ]
            expected.append(rev - ref)
        actual = r[
            (r.policy == 'reverse') & (r.metric == 'qini_score_traditional')
        ].iloc[0]
        np.testing.assert_allclose(
            [actual.vs_propensity_ci_low, actual.vs_propensity_ci_high],
            np.quantile(expected, [0.025, 0.975]),
        )

    def test_no_draws_and_invalid_bootstrap(self):
        r = ranking_confidence_intervals(self.y, self.t, {'s': self.s}, 0.5, 0)
        self.assertTrue(r.ci_low.isna().all())
        self.assertTrue(r.vs_propensity.isna().all())
        for draws in [-1, 1, 2.5]:
            with self.assertRaises(ValueError):
                ranking_confidence_intervals(self.y, self.t, {'s': self.s}, 0.5, draws)
        with self.assertRaises(ValueError):
            ranking_confidence_intervals(self.y, self.t, {}, 0.5)


if __name__ == "__main__":
    unittest.main()
