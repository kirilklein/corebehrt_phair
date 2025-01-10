import logging
import unittest

import numpy as np
import pandas as pd

from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    OUTCOME_PROBABILITY_COL,
    TREATMENT_COL,
    OUTCOME_COL,
    PID_COL,
    PROBA_COL,
)
from ehr2vec.effect_estimation.data import (
    _add_outcome_predictions,
    _assign_counterfactuals,
    construct_from_observed_data,
    _merge_prediction_column,
    _merge_and_format_outcomes,
)

TEMP_CF_COL = "Y_hat_counterfactual"
logger = logging.getLogger(__name__)


class TestConstructDataForEffectEstimation(unittest.TestCase):
    def test_construct_from_observed_data(self):
        # Set up input dataframes
        propensities = pd.DataFrame(
            {PROBA_COL: [0.2, 0.8, 0.5], TREATMENT_COL: [0, 1, 0]}, index=[1, 2, 3]
        )
        propensities.index.name = PID_COL

        outcomes = pd.DataFrame({OUTCOME_COL: [1]}, index=[1])
        outcomes.index.name = PID_COL

        # Run function
        result = construct_from_observed_data(propensities, outcomes)

        # Set up expected result
        expected_result = pd.DataFrame(
            {
                PROBA_COL: [0.2, 0.8, 0.5],
                TREATMENT_COL: [0, 1, 0],
                OUTCOME_COL: [1, 0, 0],
            },
            index=[1, 2, 3],
        )
        expected_result.index.name = PID_COL

        # Assert the final DataFrame is as expected
        pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)


class TestOutcomePredictionFunctions(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing."""
        # Create sample patient data with known treatment assignments
        self.df = pd.DataFrame(
            {
                TREATMENT_COL: [1, 0, 1, 0],  # 1=treated, 0=untreated
                OUTCOME_COL: [1, 0, 1, 0],  # actual outcomes
            },
            index=[101, 102, 103, 104],  # patient IDs
        )

        # Predicted outcomes under actual treatment
        self.outcome_predictions = pd.DataFrame(
            {
                OUTCOME_PROBABILITY_COL: [
                    0.9,
                    0.1,
                    0.8,
                    0.2,
                ],  # predictions match treatment status
            },
            index=[101, 102, 103, 104],
        )

        # Predicted outcomes under opposite treatment
        self.counterfactual_predictions = pd.DataFrame(
            {
                OUTCOME_PROBABILITY_COL: [
                    0.3,
                    0.7,
                    0.4,
                    0.6,
                ],  # predictions for opposite treatment
            },
            index=[101, 102, 103, 104],
        )

    def test_merge_prediction_column(self):
        """Test that prediction columns are merged correctly."""
        merged_df = _merge_prediction_column(
            self.df.copy(),
            self.outcome_predictions.copy(),
            OUTCOME_PROBABILITY_COL,
            "new_predictions",
        )

        # Check structure
        self.assertIn("new_predictions", merged_df.columns)
        self.assertEqual(len(merged_df), 4)
        pd.testing.assert_index_equal(merged_df.index, self.df.index)

        # Check values
        expected_predictions = [0.9, 0.1, 0.8, 0.2]
        np.testing.assert_array_almost_equal(
            merged_df["new_predictions"], expected_predictions
        )

    def test_assign_counterfactuals_basic(self):
        """Test basic counterfactual assignment logic."""
        # Prepare input DataFrame
        df = self.df.copy()
        df[OUTCOME_PROBABILITY_COL] = [0.9, 0.1, 0.8, 0.2]  # actual predictions
        df[TEMP_CF_COL] = [0.3, 0.7, 0.4, 0.6]  # counterfactual predictions

        # Run function
        result = _assign_counterfactuals(df)

        # Check treated outcomes (Y1_hat)
        # For treated patients (index 101, 103): use actual prediction
        # For untreated patients (index 102, 104): use counterfactual
        expected_treated = [0.9, 0.7, 0.8, 0.6]
        np.testing.assert_array_almost_equal(result[CF_TREATED_COL], expected_treated)

        # Check control outcomes (Y0_hat)
        # For treated patients (index 101, 103): use counterfactual
        # For untreated patients (index 102, 104): use actual prediction
        expected_control = [0.3, 0.1, 0.4, 0.2]
        np.testing.assert_array_almost_equal(result[CF_CONTROL_COL], expected_control)

    def test_assign_counterfactuals_edge_cases(self):
        """Test edge cases for counterfactual assignment."""
        # Test with extreme probabilities
        df = self.df.copy()
        df[OUTCOME_PROBABILITY_COL] = [1.0, 0.0, 1.0, 0.0]
        df[TEMP_CF_COL] = [0.0, 1.0, 0.0, 1.0]

        result = _assign_counterfactuals(df)

        # Verify extreme values are handled correctly
        np.testing.assert_array_almost_equal(
            result[CF_TREATED_COL], [1.0, 1.0, 1.0, 1.0]
        )
        np.testing.assert_array_almost_equal(
            result[CF_CONTROL_COL], [0.0, 0.0, 0.0, 0.0]
        )

    def test_assign_counterfactuals_missing_columns(self):
        """Test error handling for missing required columns."""
        # Test missing treatment column
        df_no_treatment = self.df.drop(columns=[TREATMENT_COL])
        df_no_treatment[OUTCOME_PROBABILITY_COL] = [0.9, 0.1, 0.8, 0.2]
        df_no_treatment[TEMP_CF_COL] = [0.3, 0.7, 0.4, 0.6]

        with self.assertRaises(KeyError):
            _assign_counterfactuals(df_no_treatment)

        # Test missing prediction columns
        df_no_predictions = self.df.copy()
        with self.assertRaises(KeyError):
            _assign_counterfactuals(df_no_predictions)

    def test_add_outcome_predictions_integration(self):
        """Test the full outcome prediction pipeline."""
        result = _add_outcome_predictions(
            self.df.copy(),
            self.outcome_predictions.copy(),
            self.counterfactual_predictions.copy(),
        )

        # Check all expected columns are present
        expected_columns = {
            TREATMENT_COL,
            OUTCOME_COL,
            OUTCOME_PROBABILITY_COL,
            CF_TREATED_COL,
            CF_CONTROL_COL,
        }
        self.assertTrue(expected_columns.issubset(result.columns))

        # Verify TEMP_CF_COL was removed
        self.assertNotIn(TEMP_CF_COL, result.columns)

        # Check final values
        np.testing.assert_array_almost_equal(
            result[CF_TREATED_COL], [0.9, 0.7, 0.8, 0.6]
        )
        np.testing.assert_array_almost_equal(
            result[CF_CONTROL_COL], [0.3, 0.1, 0.4, 0.2]
        )

    def test_merge_and_format_outcomes(self):
        """Test merging propensity scores with outcomes and outcome formatting."""
        # Set up test data
        propensity_scores = pd.DataFrame(
            {TREATMENT_COL: [1, 0, 1, 0], PROBA_COL: [0.7, 0.3, 0.8, 0.2]},
            index=[101, 102, 103, 104],  # patient IDs
        )

        # Only some patients have outcomes
        outcomes = pd.DataFrame(
            {OUTCOME_COL: [1, 0]}, index=[101, 103]  # only two patients have outcomes
        )

        # Run function
        result = _merge_and_format_outcomes(propensity_scores, outcomes)

        # Check structure
        self.assertEqual(
            len(result), 4
        )  # should keep all patients from propensity_scores
        self.assertTrue(
            all(
                col in result.columns for col in [TREATMENT_COL, PROBA_COL, OUTCOME_COL]
            )
        )

        # Check values
        expected_outcomes = [1, 0, 0, 0]  # missing outcomes should be 0
        np.testing.assert_array_equal(result[OUTCOME_COL], expected_outcomes)

        # Check data types
        self.assertTrue(
            np.issubdtype(result[OUTCOME_COL].dtype, np.integer)
        )  # outcomes should be integers

        # Check index preservation
        pd.testing.assert_index_equal(result.index, propensity_scores.index)


if __name__ == "__main__":
    unittest.main()
