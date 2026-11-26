import unittest

import numpy as np

from regime_pricing.projection import REVEAL, VOLS, dupire_variance, projection_demo
from regime_pricing.signal_scores import signal_price_scores


class SignalPricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        example = projection_demo()
        cls.quotes = example["quotes"]["90"]
        cls.scores = signal_price_scores(example)

    def test_local_variance_at_reveal(self):
        # At the instant of the signal the two components have the same spot law.
        weights = np.array([0.525, 0.475])
        actual = dupire_variance(REVEAL, np.array([-0.1, 0.0, 0.1]))
        np.testing.assert_allclose(actual, weights @ VOLS**2, rtol=1e-12)

    def test_conditional_and_projected_prices(self):
        for state, expected in (("calm", 1.133), ("stress", 5.169), ("dupire", 3.141)):
            with self.subTest(state=state):
                self.assertAlmostEqual(self.quotes[state]["premium"], expected, places=3)
        self.assertLess(self.quotes["grid_change"]["premium"], 0.001)

    def test_half_response_prices(self):
        self.assertEqual(self.scores["half_response_prices"], {"calm": "2.142", "stress": "4.160"})

    def test_pricing_rmse_against_analytic_targets(self):
        self.assertEqual(
            self.scores["pricing_rmse"],
            {"regime": "0.000", "half_response": "1.009", "dupire": "2.018"},
        )


if __name__ == "__main__":
    unittest.main()
