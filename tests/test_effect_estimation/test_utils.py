import unittest
import pandas as pd
from ehr2vec.effect_estimation.utils import convert_effect_to_dataframe


class TestConvertEffectToDataFrame(unittest.TestCase):
    def setUp(self):
        # Set up effect dictionary
        self.effect = {
            "IPW": {
                "effect": 0.58,
                "std_err": 0.1,
                "bootstrap": True,
                "n_bootstraps": 1000,
            },
            "TMLE": {
                "effect": 0.65,
                "std_err": 0.08,
                "bootstrap": False,
                "n_bootstraps": 0,
            },
        }

        # Set up expected result
        self.expected_result = pd.DataFrame(
            {
                "method": ["IPW", "TMLE"],
                "effect": [0.58, 0.65],
                "std_err": [0.1, 0.08],
                "bootstrap": [True, False],
                "n_bootstraps": [1000, 0],
            }
        )

    def test_convert_effect_to_dataframe(self):
        # Run function
        result = convert_effect_to_dataframe(self.effect)

        # Assert the final DataFrame is as expected, ignoring dtype
        pd.testing.assert_frame_equal(result, self.expected_result, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
