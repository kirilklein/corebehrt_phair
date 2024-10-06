import unittest
import pandas as pd
from ehr2vec.effect_estimation.data import construct_data_for_effect_estimation


class TestConstructDataForEffectEstimation(unittest.TestCase):
    def test_construct_data_for_effect_estimation(self):
        # Set up input dataframes
        propensities = pd.DataFrame(
            {"proba": [0.2, 0.8, 0.5], "treatment": [0, 1, 0]}, index=[1, 2, 3]
        )
        propensities.index.name = "PID"

        outcomes = pd.DataFrame({"outcome": [1]}, index=[1])
        outcomes.index.name = "PID"

        # Run function
        result = construct_data_for_effect_estimation(propensities, outcomes)

        # Set up expected result
        expected_result = pd.DataFrame(
            {"proba": [0.2, 0.8, 0.5], "treatment": [0, 1, 0], "outcome": [1, 0, 0]},
            index=[1, 2, 3],
        )
        expected_result.index.name = "PID"

        # Assert the final DataFrame is as expected
        pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
