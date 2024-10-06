import unittest
import pandas as pd
from ehr2vec.effect_estimation.counterfactual import compute_effect_from_counterfactuals


class TestComputeEffectFromCounterfactuals(unittest.TestCase):
    def setUp(self):
        # Set up input dataframe
        self.df = pd.DataFrame(
            {"Y1": [0.8, 0.9, 0.7], "Y0": [0.5, 0.4, 0.6], "treatment": [1, 0, 1]}
        )

    def test_ate(self):
        result = compute_effect_from_counterfactuals(self.df, "ATE")
        expected_result = self.df["Y1"].mean() - self.df["Y0"].mean()
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_att(self):
        result = compute_effect_from_counterfactuals(self.df, "ATT")
        subset = self.df[self.df["treatment"] == 1]
        expected_result = subset["Y1"].mean() - subset["Y0"].mean()
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_atc(self):
        result = compute_effect_from_counterfactuals(self.df, "ATC")
        subset = self.df[self.df["treatment"] == 0]
        expected_result = subset["Y1"].mean() - subset["Y0"].mean()
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_rr(self):
        result = compute_effect_from_counterfactuals(self.df, "RR")
        expected_result = (self.df["Y1"].mean() + 1) / (self.df["Y0"].mean() + 1)
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_or(self):
        result = compute_effect_from_counterfactuals(self.df, "OR")
        y1_mean = self.df["Y1"].mean()
        y0_mean = self.df["Y0"].mean()
        expected_result = (y1_mean / (1 - y1_mean)) / (y0_mean / (1 - y0_mean))
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_invalid_effect_type(self):
        with self.assertRaises(ValueError):
            compute_effect_from_counterfactuals(self.df, "INVALID")


if __name__ == "__main__":
    unittest.main()
