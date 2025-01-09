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
        # Sample data for testing
        self.df = pd.DataFrame(
            {
                TREATMENT_COL: [1, 0, 1, 0],
                OUTCOME_COL: [1, 0, 1, 0],
            },
            index=[101, 102, 103, 104],
        )

        self.outcome_predictions = pd.DataFrame(
            {
                OUTCOME_PROBABILITY_COL: [0.9, 0.1, 0.8, 0.2],
            },
            index=[101, 102, 103, 104],
        )

        self.counterfactual_predictions = pd.DataFrame(
            {
                OUTCOME_PROBABILITY_COL: [0.3, 0.7, 0.4, 0.6],
            },
            index=[101, 102, 103, 104],
        )

    def test__merge_prediction_column(self):
        # Test merging works correctly
        merged_df = _merge_prediction_column(
            self.df.copy(),
            self.outcome_predictions.copy(),
            OUTCOME_PROBABILITY_COL,
            OUTCOME_PROBABILITY_COL,
        )
        self.assertIn(OUTCOME_PROBABILITY_COL, merged_df.columns)
        self.assertEqual(len(merged_df), 4)
        pd.testing.assert_index_equal(merged_df.index, self.df.index)

    def test__assign_counterfactuals(self):
        # Prepare DataFrame with necessary columns
        df = self.df.copy()
        df[OUTCOME_PROBABILITY_COL] = [0.9, 0.1, 0.8, 0.2]
        df[TEMP_CF_COL] = [0.3, 0.7, 0.4, 0.6]

        # Assign counterfactuals
        df = _assign_counterfactuals(df)

        # Expected values
        expected_Y1_hat = [0.9, 0.7, 0.8, 0.6]
        expected_Y0_hat = [0.3, 0.1, 0.4, 0.2]

        np.testing.assert_array_almost_equal(df[CF_TREATED_COL], expected_Y1_hat)
        np.testing.assert_array_almost_equal(df[CF_CONTROL_COL], expected_Y0_hat)

    def test__add_outcome_predictions(self):
        # Test the main function
        df = _add_outcome_predictions(
            self.df.copy(),
            self.outcome_predictions.copy(),
            self.counterfactual_predictions.copy(),
        )

        # Check columns
        self.assertIn(OUTCOME_PROBABILITY_COL, df.columns)
        self.assertIn(TEMP_CF_COL, df.columns)
        self.assertIn(CF_TREATED_COL, df.columns)
        self.assertIn(CF_CONTROL_COL, df.columns)

        # Check assignments
        expected_Y1_hat = [0.9, 0.7, 0.8, 0.6]
        expected_Y0_hat = [0.3, 0.1, 0.4, 0.2]

        np.testing.assert_array_almost_equal(df[CF_TREATED_COL], expected_Y1_hat)
        np.testing.assert_array_almost_equal(df[CF_CONTROL_COL], expected_Y0_hat)

    def test_non_matching_indices(self):
        # Change indices so they don't match
        outcome_predictions_mismatch = self.outcome_predictions.copy()
        outcome_predictions_mismatch.index = [201, 202, 203, 204]

        df_result = _add_outcome_predictions(
            self.df.copy(),
            outcome_predictions_mismatch,
            self.counterfactual_predictions.copy(),
        )

        self.assertEqual(len(df_result), 0)

    def test_empty_dataframes(self):
        # Test with empty DataFrames
        empty_df = pd.DataFrame(columns=self.df.columns)
        empty_outcome_predictions = pd.DataFrame(
            columns=self.outcome_predictions.columns
        )
        empty_counterfactual_predictions = pd.DataFrame(
            columns=self.counterfactual_predictions.columns
        )

        df_result = _add_outcome_predictions(
            empty_df, empty_outcome_predictions, empty_counterfactual_predictions
        )

        self.assertEqual(len(df_result), 0)
        self.assertEqual(
            list(df_result.columns),
            list(self.df.columns)
            + [
                OUTCOME_PROBABILITY_COL,
                TEMP_CF_COL,
                CF_TREATED_COL,
                CF_CONTROL_COL,
            ],
        )

    def test_partial_overlap_indices(self):
        # Modify indices to have partial overlap
        outcome_predictions_partial = self.outcome_predictions.copy()
        outcome_predictions_partial.index = [101, 102, 201, 202]

        df_result = _add_outcome_predictions(
            self.df.copy(),
            outcome_predictions_partial,
            self.counterfactual_predictions.copy(),
        )

        # Only indices 101 and 102 should be present
        self.assertEqual(len(df_result), 2)
        self.assertListEqual(list(df_result.index), [101, 102])

    def test_incorrect_treatment_column(self):
        # Missing TREATMENT_COL in df
        df_missing_treatment = self.df.drop(columns=[TREATMENT_COL])

        with self.assertRaises(KeyError):
            _assign_counterfactuals(df_missing_treatment)

    def test_incorrect_prediction_columns(self):
        # Missing OUTCOME_PREDICTIONS_COL in df
        df_missing_prediction = self.df.copy()
        df_missing_prediction[TEMP_CF_COL] = [0.3, 0.7, 0.4, 0.6]

        with self.assertRaises(KeyError):
            _assign_counterfactuals(df_missing_prediction)

    def test__assign_counterfactuals_with_nonbinary_treatment(self):
        # Non-binary treatment values
        df_nonbinary_treatment = self.df.copy()
        df_nonbinary_treatment[TREATMENT_COL] = [2, -1, 1, 0]
        df_nonbinary_treatment[OUTCOME_PROBABILITY_COL] = [0.9, 0.1, 0.8, 0.2]
        df_nonbinary_treatment[TEMP_CF_COL] = [0.3, 0.7, 0.4, 0.6]

        df_result = _assign_counterfactuals(df_nonbinary_treatment)

        # Treated if TREATMENT_COL == 1
        expected_Y1_hat = [0.3, 0.7, 0.8, 0.6]
        expected_Y0_hat = [0.9, 0.1, 0.4, 0.2]

        np.testing.assert_array_almost_equal(df_result[CF_TREATED_COL], expected_Y1_hat)
        np.testing.assert_array_almost_equal(df_result[CF_CONTROL_COL], expected_Y0_hat)


if __name__ == "__main__":
    unittest.main()
