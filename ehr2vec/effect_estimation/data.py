import logging

import numpy as np
import pandas as pd

from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    OUTCOME_PROBABILITY_COL,
    TREATMENT_COL,
    OUTCOME_COL,
)
from ehr2vec.data.utils import remove_duplicate_indices

TEMP_CF_COL = "Y_hat_counterfactual"

logger = logging.getLogger(__name__)


def construct_from_observed_data(
    propensity_scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_predictions: pd.DataFrame = None,
    counterfactual_predictions: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Constructs the data for causal effect estimation by merging propensity scores and outcomes.

    Args:
        propensity_scores: DataFrame with patient IDs as index and columns for propensity scores
            and treatment status
        outcomes: DataFrame with patient IDs as index containing outcome timestamps. If a patient
            is not present in this DataFrame, their outcome is set to 0
        outcome_predictions: Optional DataFrame with patient IDs as index containing predicted
            outcomes under observed treatment
        counterfactual_predictions: Optional DataFrame with patient IDs as index containing predicted
            outcomes under counterfactual treatment

    Returns:
        DataFrame with patient IDs as index and columns for:
            - Propensity scores ('proba')
            - Treatment status ('treatment')
            - Binary outcome ('outcome')
            - Predicted outcomes if outcome_predictions provided
            - Counterfactual predictions if counterfactual_predictions provided

    Note:
        The returned DataFrame will only contain patients present in the propensity_scores DataFrame.
        Missing outcomes are filled with 0 and cast to integer type.
    """
    df = _merge_and_format_outcomes(propensity_scores, outcomes)

    if outcome_predictions is not None and counterfactual_predictions is not None:
        df = _add_outcome_predictions(
            df=df,
            outcome_predictions=outcome_predictions,
            counterfactual_predictions=counterfactual_predictions,
        )

    return df


def _merge_and_format_outcomes(
    propensity_scores: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Merges propensity scores with outcomes and formats outcome values.

    Performs a left merge of propensity scores with outcomes, keeping all patients from
    propensity scores. Missing outcomes are filled with 0 and cast to integer type.

    Args:
        propensity_scores: DataFrame with propensity scores and treatment status, indexed by patient ID
        outcomes: DataFrame with binary outcomes, indexed by patient ID

    Returns:
        DataFrame containing propensity scores and formatted binary outcomes (0/1)
    """
    df = pd.merge(
        propensity_scores, outcomes, left_index=True, right_index=True, how="left"
    )
    df.loc[:, OUTCOME_COL] = df[OUTCOME_COL].fillna(0)
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    return df


def construct_from_counterfactuals(
    propensity_scores: pd.DataFrame, counterfactual_outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Constructs data for causal effect estimation by merging propensity scores with counterfactual outcomes.

    Takes propensity scores and counterfactual outcomes and merges them into a single DataFrame
    for causal effect estimation. The counterfactual outcomes contain the potential outcomes
    under treatment (Y1) and control (Y0) for each patient.

    Args:
        propensity_scores: DataFrame containing propensity scores indexed by patient ID
        counterfactual_outcomes: DataFrame containing Y1 and Y0 columns with patient ID in 'PID' column

    Returns:
        pd.DataFrame: Merged DataFrame containing propensity scores and counterfactual outcomes Y1 and Y0,
            only including patients present in both input DataFrames

    Note:
        - Sets PID as index on counterfactual_outcomes before merging
        - Performs inner join to only keep patients present in both DataFrames
        - Validates 1:1 relationship between DataFrames during merge
    """
    counterfactual_outcomes = counterfactual_outcomes.set_index("PID")
    df = pd.merge(
        propensity_scores,
        counterfactual_outcomes,
        left_index=True,
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    return df


def _add_outcome_predictions(
    df: pd.DataFrame,
    outcome_predictions: pd.DataFrame,
    counterfactual_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds outcome predictions and counterfactual predictions to the input DataFrame.

    This function merges the input DataFrame with outcome predictions and counterfactual predictions,
    assigns counterfactual outcomes based on treatment status, and performs data integrity checks.

    Args:
        df: Input DataFrame containing treatment and outcome information.
        outcome_predictions: DataFrame with outcome predictions.
        counterfactual_predictions: DataFrame with counterfactual predictions.

    Returns:
        df: Updated DataFrame with added outcome and counterfactual predictions.

    Note:
        - This function removes duplicate indices from all input DataFrames.
        - It logs warnings if the number of unique PIDs is reduced during merging.
        - The function assigns Y1_hat and Y0_hat based on the treatment status.
    """
    df = remove_duplicate_indices(df)
    outcome_predictions = remove_duplicate_indices(outcome_predictions)
    counterfactual_predictions = remove_duplicate_indices(counterfactual_predictions)

    initial_pids = df.index.unique()

    df = _merge_prediction_column(
        df, outcome_predictions, OUTCOME_PROBABILITY_COL, OUTCOME_PROBABILITY_COL
    )

    if len(df.index.unique()) != len(initial_pids):
        logger.warning(
            f"Number of unique PIDs reduced from {len(initial_pids)} to {len(df.index.unique())}"
        )

    df = _merge_prediction_column(
        df, counterfactual_predictions, OUTCOME_PROBABILITY_COL, TEMP_CF_COL
    )
    if len(df.index.unique()) != len(initial_pids):
        logger.warning(
            f"Number of unique PIDs reduced from {len(initial_pids)} to {len(df.index.unique())}"
        )

    df = _assign_counterfactuals(df)
    df.drop(columns=[TEMP_CF_COL], inplace=True)

    logger.info(f"Final DataFrame shape: {df.shape}, Unique PIDs: {df.index.nunique()}")

    return df


def _merge_prediction_column(
    df: pd.DataFrame, predictions: pd.DataFrame, predictions_col: str, new_col_name: str
) -> pd.DataFrame:
    """Merge a predictions DataFrame into the main DataFrame by matching their indices.

    This function takes a predictions DataFrame and merges a single column of predictions
    into the main DataFrame. It performs an inner merge, meaning only rows with matching
    indices in both DataFrames are kept. The predictions column is renamed before merging
    and all other columns in the predictions DataFrame are discarded.

    Args:
        df: Main DataFrame that predictions will be merged into. Must have an index that
            matches with the predictions DataFrame.
        predictions: DataFrame containing the prediction values to merge. Must have an index
            that matches with the main DataFrame.
        predictions_col: Name of the column in predictions DataFrame that contains the actual
            prediction values to merge.
        new_col_name: The name to give the predictions column in the merged DataFrame. The
            predictions will appear under this name in the output.

    Returns:
        pd.DataFrame: A new DataFrame containing all columns from the input df plus the
            predictions column under new_col_name. Only contains rows where indices matched
            between df and predictions.

    Note:
        - Uses an inner merge strategy, so rows will be dropped if their indices don't exist
          in both DataFrames
        - Only the specified predictions column is merged, all other columns in the predictions
          DataFrame are ignored
        - The original DataFrames are not modified - a new merged DataFrame is returned
    """
    predictions = predictions.rename(columns={predictions_col: new_col_name})
    return df.merge(
        predictions[[new_col_name]], left_index=True, right_index=True, how="inner"
    )


def _assign_counterfactuals(df: pd.DataFrame) -> pd.DataFrame:
    """Assign counterfactual outcome predictions based on treatment status.

    For each patient, assigns their predicted outcomes under treatment (Y1_hat) and
    control (Y0_hat) conditions. For treated patients, Y1_hat is their actual predicted
    outcome and Y0_hat is their counterfactual prediction. For untreated patients,
    Y0_hat is their actual predicted outcome and Y1_hat is their counterfactual prediction.

    Args:
        df: DataFrame containing treatment status (TREATMENT_COL), predicted outcomes
            (OUTCOME_PROBABILITY_COL), and counterfactual predictions (TEMP_CF_COL)

    Returns:
        DataFrame with additional columns for predicted outcomes under treatment
        (CF_TREATED_COL) and control (CF_CONTROL_COL)
    """
    is_treated = df[TREATMENT_COL] == 1
    is_untreated = ~is_treated

    actual_outcomes = df[OUTCOME_PROBABILITY_COL]
    cf_outcomes = df[TEMP_CF_COL]

    df[CF_TREATED_COL] = np.where(is_treated, actual_outcomes, cf_outcomes)
    df[CF_CONTROL_COL] = np.where(is_untreated, actual_outcomes, cf_outcomes)
    return df
