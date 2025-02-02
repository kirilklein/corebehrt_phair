import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    OUTCOME_PROBABILITY_COL,
    PID_COL,
    PS_COL,
    TREATMENT_COL,
)
from ehr2vec.effect_estimation.main_estimator import (
    EffectEstimator,
    EffectEstimator_with_transform,
    DELTA_PS,
    DELTA_Y,
    DELTA_CF,
    CONSTANTS,
)


class TestEffectEstimator(unittest.TestCase):
    def setUp(self):
        self.mock_cfg = Mock()
        self.mock_cfg.paths = Mock()
        self.mock_logger = Mock()
        self.mock_mount_context = Mock()
        self.estimator = EffectEstimator(
            cfg=self.mock_cfg,
            logger=self.mock_logger,
            exp_folder="test_folder",
            mount_context=self.mock_mount_context,
        )

    def test_load_propensity_scores(self):
        # Create test data
        test_data = pd.DataFrame(
            {"pid": [1, 2, 3], "target": [0, 1, 0], "proba": [0.1, 0.8, 0.3]}
        )

        with patch("pandas.read_csv", return_value=test_data):
            result = self.estimator._load_propensity_scores("dummy_path")

            self.assertEqual(list(result.index), [1, 2, 3])
            self.assertEqual(list(result[TREATMENT_COL]), [0, 1, 0])
            self.assertEqual(list(result[PS_COL]), [0.1, 0.8, 0.3])

    def test_load_predictions(self):
        test_data = pd.DataFrame({"pid": [1, 2, 3], "proba": [0.2, 0.5, 0.7]})

        with patch("pandas.read_csv", return_value=test_data):
            result = self.estimator._load_predictions("dummy_path")

            self.assertEqual(list(result.index), [1, 2, 3])
            self.assertEqual(list(result[OUTCOME_PROBABILITY_COL]), [0.2, 0.5, 0.7])

    def test_sample_patients(self):
        test_df = pd.DataFrame(
            {PID_COL: range(100), TREATMENT_COL: np.random.binomial(1, 0.5, 100)}
        )

        # Test without sampling
        self.mock_cfg.get.return_value = False
        result = self.estimator._sample_patients(test_df)
        self.assertEqual(len(result), 100)

        # Test with sampling
        self.mock_cfg.get.return_value = 50
        result = self.estimator._sample_patients(test_df)
        self.assertEqual(len(result), 50)


class TestEffectEstimatorWithTransform(unittest.TestCase):
    def setUp(self):
        self.mock_cfg = Mock()
        self.mock_logger = Mock()
        self.mock_mount_context = Mock()
        self.estimator = EffectEstimator_with_transform(
            cfg=self.mock_cfg,
            logger=self.mock_logger,
            exp_folder="test_folder",
            mount_context=self.mock_mount_context,
        )

    def test_add_bias(self):
        probas = pd.Series([0.2, 0.5, 0.8])
        constants = {"sigma": 0}  # No random noise

        # Test positive bias
        result = self.estimator._add_bias(probas, delta=1.0, constants=constants)
        self.assertTrue(all(result > probas))  # All values should increase

        # Test negative bias
        result = self.estimator._add_bias(probas, delta=-1.0, constants=constants)
        self.assertTrue(all(result < probas))  # All values should decrease

        # Test zero bias
        result = self.estimator._add_bias(probas, delta=0.0, constants=constants)
        np.testing.assert_array_almost_equal(result, probas)

    def test_power_distortion(self):
        probas = pd.Series([0.2, 0.5, 0.8])
        constants = {}

        # Test sharpening (alpha > 1)
        result = self.estimator._power_distortion(
            probas, alpha=2.0, constants=constants
        )
        self.assertTrue(all((result < probas) == (probas < 0.5)))

        # Test flattening (alpha < 1)
        result = self.estimator._power_distortion(
            probas, alpha=0.5, constants=constants
        )
        self.assertTrue(all((result > probas) == (probas < 0.5)))

        # Test no distortion (alpha = 1)
        result = self.estimator._power_distortion(
            probas, alpha=1.0, constants=constants
        )
        np.testing.assert_array_almost_equal(result, probas)

    def test_transform_data(self):
        df = pd.DataFrame(
            {
                PS_COL: [0.2, 0.5, 0.8],
                OUTCOME_PROBABILITY_COL: [0.3, 0.6, 0.7],
                CF_TREATED_COL: [0.4, 0.5, 0.6],
                CF_CONTROL_COL: [0.2, 0.3, 0.4],
            }
        )

        params = {DELTA_PS: 1.0, DELTA_Y: 0.5, DELTA_CF: 0.5, CONSTANTS: {"sigma": 0}}

        result = self.estimator._transform_data(
            df.copy(), params, self.estimator._add_bias
        )

        # Check that all columns were transformed
        self.assertTrue(all(result[PS_COL] != df[PS_COL]))
        self.assertTrue(
            all(result[OUTCOME_PROBABILITY_COL] != df[OUTCOME_PROBABILITY_COL])
        )
        self.assertTrue(all(result[CF_TREATED_COL] != df[CF_TREATED_COL]))
        self.assertTrue(all(result[CF_CONTROL_COL] != df[CF_CONTROL_COL]))

    def test_get_transform_function(self):
        # Test add_bias function selection
        self.mock_cfg.get.return_value = "add_bias"
        func = self.estimator.get_transform_function()
        self.assertEqual(func, self.estimator._add_bias)

        # Test power_distortion function selection
        self.mock_cfg.get.return_value = "power_distortion"
        func = self.estimator.get_transform_function()
        self.assertEqual(func, self.estimator._power_distortion)

        # Test invalid function name
        self.mock_cfg.get.return_value = "invalid_function"
        with self.assertRaises(ValueError):
            self.estimator.get_transform_function()

    def test_initialize_results(self):
        # Test with simple parameters
        params = {
            DELTA_PS: 1.0,
            DELTA_Y: 0.5,
            DELTA_CF: 0.3,
            CONSTANTS: {"sigma": 0.1, "alpha": 2.0},
        }

        results = self.estimator._initialize_results(params)

        # Check that all parameters are initialized as empty lists
        self.assertEqual(results[DELTA_PS], [])
        self.assertEqual(results[DELTA_Y], [])
        self.assertEqual(results[DELTA_CF], [])
        self.assertEqual(results["sigma"], [])
        self.assertEqual(results["alpha"], [])

        # Test with empty constants
        params_no_constants = {
            DELTA_PS: 1.0,
            DELTA_Y: 0.5,
            DELTA_CF: 0.3,
            CONSTANTS: {},
        }

        results = self.estimator._initialize_results(params_no_constants)
        self.assertEqual(list(results.keys()), [DELTA_PS, DELTA_Y, DELTA_CF])

    def test_append_to_results(self):
        # Initialize results dict
        results = {DELTA_PS: [], DELTA_Y: [], DELTA_CF: [], "sigma": [], "alpha": []}

        # Test appending single set of parameters
        params = {
            DELTA_PS: 1.0,
            DELTA_Y: 0.5,
            DELTA_CF: 0.3,
            CONSTANTS: {"sigma": 0.1, "alpha": 2.0},
        }

        results = self.estimator._append_to_results(results, params)

        # Check that values were appended correctly
        self.assertEqual(results[DELTA_PS], [1.0])
        self.assertEqual(results[DELTA_Y], [0.5])
        self.assertEqual(results[DELTA_CF], [0.3])
        self.assertEqual(results["sigma"], [0.1])
        self.assertEqual(results["alpha"], [2.0])

        # Test appending another set of parameters
        params2 = {
            DELTA_PS: 0.8,
            DELTA_Y: 0.6,
            DELTA_CF: 0.4,
            CONSTANTS: {"sigma": 0.2, "alpha": 1.5},
        }

        results = self.estimator._append_to_results(results, params2)

        # Check that values were appended correctly
        self.assertEqual(results[DELTA_PS], [1.0, 0.8])
        self.assertEqual(results[DELTA_Y], [0.5, 0.6])
        self.assertEqual(results[DELTA_CF], [0.3, 0.4])
        self.assertEqual(results["sigma"], [0.1, 0.2])
        self.assertEqual(results["alpha"], [2.0, 1.5])

    def test_append_effect_to_results(self):
        # Initialize results dict
        results = {}

        # Create mock effect dataframe
        effect_df = pd.DataFrame(
            {
                "method": ["AIPW", "TMLE"],
                "effect": [0.25, 0.30],
                "std_err": [0.05, 0.06],
            }
        )

        results = self.estimator._append_effect_to_results(results, effect_df)

        # Check that effect and std_err were added for each method
        self.assertEqual(results["effect_AIPW"], [0.25])
        self.assertEqual(results["effect_std_AIPW"], [0.05])
        self.assertEqual(results["effect_TMLE"], [0.30])
        self.assertEqual(results["effect_std_TMLE"], [0.06])

        # Test appending another set of effects
        effect_df2 = pd.DataFrame(
            {
                "method": ["AIPW", "TMLE"],
                "effect": [0.35, 0.40],
                "std_err": [0.07, 0.08],
            }
        )

        results = self.estimator._append_effect_to_results(results, effect_df2)

        # Check that new effects were appended correctly
        self.assertEqual(results["effect_AIPW"], [0.25, 0.35])
        self.assertEqual(results["effect_std_AIPW"], [0.05, 0.07])
        self.assertEqual(results["effect_TMLE"], [0.30, 0.40])
        self.assertEqual(results["effect_std_TMLE"], [0.06, 0.08])

    def test_results_alignment(self):
        # Create test parameters
        test_params = [
            {DELTA_PS: 0, DELTA_Y: 0, DELTA_CF: 0, CONSTANTS: {"sigma": 0}},
            {DELTA_PS: 1, DELTA_Y: 1, DELTA_CF: 1, CONSTANTS: {"sigma": 0}},
        ]

        # Initialize results
        results = self.estimator._initialize_results(test_params[0])
        for param in test_params:
            results = self.estimator._append_to_results(results, param)

        # Verify all arrays have same length
        lengths = {k: len(v) for k, v in results.items()}
        assert len(set(lengths.values())) == 1
        assert list(lengths.values())[0] == len(test_params)


if __name__ == "__main__":
    unittest.main()
