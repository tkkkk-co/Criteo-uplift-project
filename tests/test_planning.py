import math
from statistics import NormalDist
import unittest
from criteo_uplift.planning import plan_policy_ab, cost_value_scenario


class PlanningTests(unittest.TestCase):
    def test_equal_arm_textbook_formula_and_power(self):
        r = plan_policy_ab(0.003, 0.0003)
        z = NormalDist()
        expected = math.ceil(
            (
                z.inv_cdf(0.975) * math.sqrt(2 * 0.00315 * 0.99685)
                + z.inv_cdf(0.8) * math.sqrt(0.003 * 0.997 + 0.0033 * 0.9967)
            )
            ** 2
            / 0.0003**2
        )
        self.assertEqual(r['n_policy_B'], expected)
        self.assertEqual(r['n_policy_A'], expected)
        self.assertGreaterEqual(r['approximate_achieved_power'], 0.8)
        self.assertLess(r['approximate_achieved_power'], 0.801)

    def test_mde_and_allocation_and_budget_semantics(self):
        r = plan_policy_ab(0.003, 0.0003, allocation_ratio=2)
        self.assertEqual(r['n_policy_A'], 2 * r['n_policy_B'])
        full = plan_policy_ab(0.003, 0.0003, budget=1)
        partial = plan_policy_ab(0.003, 0.0003, budget=0.2)
        self.assertEqual(full['n_total'], partial['n_total'])
        self.assertGreater(
            plan_policy_ab(0.003, 0.00015)['n_total'], 3 * partial['n_total']
        )

    def test_economics_hand_example_and_ci(self):
        r = cost_value_scenario(
            100,
            budget=0.2,
            value_per_conversion=10,
            contact_cost=0.02,
            ci_low=50,
            ci_high=150,
        )
        self.assertEqual(r['contacts'], 20000)
        self.assertEqual(r['net_incremental_value'], 600)
        self.assertEqual(r['break_even_contact_cost'], 0.05)
        self.assertEqual(r['net_ci_low_conditional'], 100)
        self.assertEqual(r['net_ci_high_conditional'], 1100)
        negative = cost_value_scenario(
            -5, budget=0.2, value_per_conversion=10, contact_cost=0
        )
        self.assertLess(negative['break_even_contact_cost'], 0)

    def test_reject_invalid_assumptions(self):
        for p, d in [
            (0, 0.01),
            (0.99, 0.02),
            (0.2, 0),
            (0.2, -0.1),
            (0.2, float('nan')),
        ]:
            with self.assertRaises(ValueError):
                plan_policy_ab(p, d)
        with self.assertRaises(ValueError):
            cost_value_scenario(
                100, budget=0.2, value_per_conversion=10, contact_cost=-1
            )
        with self.assertRaises(ValueError):
            cost_value_scenario(
                100, budget=0.2, value_per_conversion=10, contact_cost=0, ci_low=1
            )
        with self.assertRaises(ValueError):
            cost_value_scenario(
                100,
                budget=0.2,
                value_per_conversion=10,
                contact_cost=0,
                ci_low=2,
                ci_high=1,
            )
