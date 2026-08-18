import unittest

from shop_env import reward


class RewardModeTests(unittest.TestCase):
    def test_strict_is_the_multiplicative_shop_simulator_reward(self):
        detail = {
            "r_type": 1.0,
            "r_att": 0.5,
            "r_option": 0.75,
            "r_price": 1.0,
        }
        self.assertAlmostEqual(
            reward.shaped(detail, budget_mode="strict"), 0.375
        )

    def test_strict_zeroes_over_budget_without_partial_credit(self):
        detail = {
            "r_type": 1.0,
            "r_att": 1.0,
            "r_option": 1.0,
            "r_price": 0.0,
        }
        self.assertEqual(reward.shaped(detail, budget_mode="strict"), 0.0)
        self.assertEqual(reward.shaped({}, budget_mode="strict"), 0.0)


if __name__ == "__main__":
    unittest.main()
