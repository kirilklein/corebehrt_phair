import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ehr2vec.common.config import Config
from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    OUTCOME_PROBABILITY_COL,
    PID_COL,
    PS_COL,
    TREATMENT_COL,
)
from ehr2vec.effect_estimation.main_estimator import (
    CONSTANTS,
    DELTA_CF,
    DELTA_PS,
    DELTA_Y,
    EffectEstimator,
    EffectEstimator_with_transform,
)
from tests.common.simulate import create_synthetic_test_data


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

    @patch.object(EffectEstimator, "_load_data")
    @patch.object(EffectEstimator_with_transform, "_load_data")
    def test_no_transform_params_produces_same_results(
        self, mock_load_data_transform, mock_load_data_standard
    ):
        """
        Check that if the transformation parameters (delta_ps, delta_y, etc.) are zero,
        the effect results match those from the standard estimator within a small tolerance.
        """
        # 1. Create synthetic test data
        df_synthetic = create_synthetic_test_data(10_000)
        COMMON_SUPPORT_THRESHOLD = 0.005
        N_BOOTSTRAP = 50
        # 2. Mock both _load_data() calls to return the same DataFrame
        mock_load_data_transform.return_value = df_synthetic.copy()
        mock_load_data_standard.return_value = df_synthetic.copy()

        # 3. Configure the standard estimator
        #    We only need to ensure that it doesn't apply transformations:
        #    The standard estimator has no transform logic by default.
        #    We'll also ensure it has the same methods, e.g., AIPW and TMLE.
        self.mock_cfg = Config(
            {
                "estimator": {
                    "methods": ["AIPW", "TMLE"],
                    "effect_type": "ATE",
                    "common_support_threshold": COMMON_SUPPORT_THRESHOLD,
                    "n_bootstrap": N_BOOTSTRAP,
                }
            }
        )

        standard_estimator = EffectEstimator(
            cfg=self.mock_cfg,
            logger=self.mock_logger,
            exp_folder="test_folder",
            mount_context=self.mock_mount_context,
        )

        # 4. Configure the transform estimator with zero deltas
        #    We need to ensure transform_function is "add_bias" (or whichever) but with zero parameters.
        self.mock_cfg = Config(
            {
                "transform_function": "add_bias",
                "transform_params": {
                    DELTA_PS: [0],  # zero bias
                    DELTA_Y: [0],  # zero bias
                    "cf_offset": 0,
                    CONSTANTS: {"sigma": 0},  # no noise
                },
                "estimator": {
                    "methods": ["AIPW", "TMLE"],
                    "effect_type": "ATE",
                    "common_support_threshold": COMMON_SUPPORT_THRESHOLD,
                    "n_bootstrap": N_BOOTSTRAP,
                },
            }
        )
        transform_estimator = EffectEstimator_with_transform(
            cfg=self.mock_cfg,
            logger=self.mock_logger,
            exp_folder="test_folder",
            mount_context=self.mock_mount_context,
        )

        # 5. Call the effect computation methods directly (bypassing .run())
        df_standard = standard_estimator._load_data()
        effect_df_standard, cs_std, thresh_std = (
            standard_estimator._compute_causal_effect(df_standard)
        )

        df_transform = transform_estimator._load_data()
        effect_df_transform, cs_trans, thresh_trans = (
            transform_estimator._compute_causal_effect(df_transform)
        )

        # 6. Compare the effect estimates
        # Pivot or index by method for easy comparison
        standard_effects = effect_df_standard.set_index("method")["effect"]
        standard_stds = effect_df_standard.set_index("method")["std_err"]
        transform_effects = effect_df_transform.set_index("method")["effect"]
        transform_stds = effect_df_transform.set_index("method")["std_err"]

        for method in standard_effects.index:
            e_std = standard_effects[method]
            e_trans = transform_effects[method]
            # Check they are approximately equal
            self.assertAlmostEqual(
                e_std,
                e_trans,
                delta=np.sqrt(standard_stds[method] ** 2 + transform_stds[method] ** 2),
                msg=f"Effects differ for method {method}: standard={e_std}, transform={e_trans}",
            )

        # Optionally compare standard errors too
        standard_stds = effect_df_standard.set_index("method")["std_err"]
        transform_stds = effect_df_transform.set_index("method")["std_err"]
        for method in standard_stds.index:
            s_std = standard_stds[method]
            s_trans = transform_stds[method]
            self.assertAlmostEqual(
                s_std,
                s_trans,
                delta=np.sqrt(standard_stds[method] ** 2 + transform_stds[method] ** 2),
                msg=f"Std errs differ for method {method}: standard={s_std}, transform={s_trans}",
            )

        # If you want, also ensure the common support flags or thresholds are the same
        self.assertEqual(cs_std, cs_trans, "Common support flags do not match.")
        self.assertEqual(
            thresh_std, thresh_trans, "Common support thresholds do not match."
        )

    @patch.object(EffectEstimator_with_transform, "_load_data")
    def test_ipw_invariant_to_outcome_transform(self, mock_load_data):
        """
        Test that IPW estimates remain unchanged when only outcome probabilities
        are transformed (delta_ps = 0, but delta_y and cf_offset ≠ 0).
        This is because IPW only depends on propensity scores.
        """
        COMMON_SUPPORT_THRESHOLD = 0.005
        N_BOOTSTRAP = 50
        # Create synthetic test data
        df_synthetic = create_synthetic_test_data(10_000)
        mock_load_data.return_value = df_synthetic.copy()

        # Base config with IPW only
        base_config = Config(
            {
                "transform_function": "add_bias",
                "estimator": {
                    "methods": ["IPW"],
                    "effect_type": "ATE",
                    "common_support_threshold": COMMON_SUPPORT_THRESHOLD,
                    "n_bootstrap": N_BOOTSTRAP,
                },
            }
        )

        # First estimator with all zeros
        base_config.transform_params = {
            DELTA_PS: [0],
            DELTA_Y: [0],
            "cf_offset": 0,
            CONSTANTS: {"sigma": 0},
        }

        estimator_zero = EffectEstimator_with_transform(
            cfg=base_config,
            logger=self.mock_logger,
            exp_folder="test_folder",
            mount_context=self.mock_mount_context,
        )

        # Second estimator with non-zero outcome transforms
        base_config.transform_params = {
            DELTA_PS: [0],  # keep PS unchanged
            DELTA_Y: [1.0],  # non-zero outcome bias
            "cf_offset": 1.5,  # non-zero counterfactual bias
            CONSTANTS: {"sigma": 0},
        }

        estimator_nonzero = EffectEstimator_with_transform(
            cfg=base_config,
            logger=self.mock_logger,
            exp_folder="test_folder",
            mount_context=self.mock_mount_context,
        )

        # Compute effects
        effect_df_zero, _, _ = estimator_zero._compute_causal_effect(df_synthetic)
        effect_df_nonzero, _, _ = estimator_nonzero._compute_causal_effect(df_synthetic)

        # Compare IPW effects
        ipw_effect_zero = effect_df_zero.set_index("method").loc["IPW", "effect"]
        ipw_std_zero = effect_df_zero.set_index("method").loc["IPW", "std_err"]
        ipw_effect_nonzero = effect_df_nonzero.set_index("method").loc["IPW", "effect"]
        ipw_std_nonzero = effect_df_nonzero.set_index("method").loc["IPW", "std_err"]

        self.assertAlmostEqual(
            ipw_effect_zero,
            ipw_effect_nonzero,
            delta=np.sqrt(ipw_std_zero**2 + ipw_std_nonzero**2),
            msg="IPW effects differ despite unchanged propensity scores",
        )

        # Compare standard errors
        ipw_std_zero = effect_df_zero.set_index("method").loc["IPW", "std_err"]
        ipw_std_nonzero = effect_df_nonzero.set_index("method").loc["IPW", "std_err"]
        self.assertAlmostEqual(
            ipw_std_zero,
            ipw_std_nonzero,
            delta=np.sqrt(ipw_std_zero**2 + ipw_std_nonzero**2),
            msg="IPW standard errors differ despite unchanged propensity scores",
        )


if __name__ == "__main__":
    unittest.main()
