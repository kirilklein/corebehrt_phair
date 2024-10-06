import unittest
import pandas as pd
from ehr2vec.effect_estimation.data import (
    construct_data_to_estimate_effect_from_counterfactuals,
)


class TestConstructDataToEstimateEffectFromCounterfactuals(unittest.TestCase):
    def setUp(self):
        # Set up input dataframes
        self.propensity_scores = pd.DataFrame(
            {"PID": [1, 2, 3], "treatment": [0, 1, 0]}
        ).set_index("PID")

        self.counterfactual_outcomes = pd.DataFrame(
            {"PID": [1, 2, 3], "Y0": [0.4, 0.5, 0.3], "Y1": [0.7, 0.8, 0.6]}
        )

        # Set up expected result
        self.expected_result = pd.DataFrame(
            {"treatment": [0, 1, 0], "Y0": [0.4, 0.5, 0.3], "Y1": [0.7, 0.8, 0.6]},
            index=[1, 2, 3],
        )
        self.expected_result.index.name = "PID"

    def test_construct_data_to_estimate_effect_from_counterfactuals(self):
        # Run function
        result = construct_data_to_estimate_effect_from_counterfactuals(
            self.propensity_scores, self.counterfactual_outcomes
        )

        # Assert the final DataFrame is as expected, ignoring dtype
        pd.testing.assert_frame_equal(result, self.expected_result, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
