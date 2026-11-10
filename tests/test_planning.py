import math
from statistics import NormalDist
import unittest
from criteo_uplift.planning import plan_policy_ab, cost_value_scenario


#-----------------------------------------------------------------------------------------------------------------------
# 类：PlanningTests —— 验证样本量与成本收益规划。
# 输入：
#     配置：由初始化方法接收；无自定义初始化方法时使用父类约定。
# 输出：
#     实例：保存测试环境和断言的对象。
# 说明：验证样本量与成本收益规划。
#-----------------------------------------------------------------------------------------------------------------------
class PlanningTests(unittest.TestCase):
    def test_equal_arm_textbook_formula_and_power(self):
        """
        函数名：PlanningTests.test_equal_arm_textbook_formula_and_power
        功能：验证等比例分组下 A/B 样本量公式及目标功效的计算结果。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
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
        """
        函数名：PlanningTests.test_mde_and_allocation_and_budget_semantics
        功能：验证 MDE、非等比例分配和触达预算参数对实验规划输出的影响。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
        r = plan_policy_ab(0.003, 0.0003, allocation_ratio=2)
        self.assertEqual(r['n_policy_A'], 2 * r['n_policy_B'])
        full = plan_policy_ab(0.003, 0.0003, budget=1)
        partial = plan_policy_ab(0.003, 0.0003, budget=0.2)
        self.assertEqual(full['n_total'], partial['n_total'])
        self.assertGreater(
            plan_policy_ab(0.003, 0.00015)['n_total'], 3 * partial['n_total']
        )

    def test_economics_hand_example_and_ci(self):
        """
        函数名：PlanningTests.test_economics_hand_example_and_ci
        功能：使用手工情景核对增量转化、收益、触达成本、净值和条件性区间。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
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
        """
        函数名：PlanningTests.test_reject_invalid_assumptions
        功能：验证样本量规划和经济情景函数拒绝越界、非有限或不完整的假设参数。
        输入参数：
            - 无显式业务参数（实例方法中的 self 表示当前对象）。
        输出参数：
            - 无；通过断言验证预期行为，失败时由测试框架报告。
        """
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
