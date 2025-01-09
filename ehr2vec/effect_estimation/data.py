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
    # Perform an outer merge but only keep PIDs in propensities
    df = pd.merge(
        propensity_scores, outcomes, left_index=True, right_index=True, how="left"
    )
    df.loc[:, OUTCOME_COL] = df[OUTCOME_COL].fillna(0)
    df[OUTCOME_COL] = df[OUTCOME_COL].astype(int)
    if counterfactual_predictions is not None and outcome_predictions is not None:
        df = _add_outcome_predictions(
            df, outcome_predictions, counterfactual_predictions
        )

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

    df = _merge_with_predictions(
        df, outcome_predictions, OUTCOME_PROBABILITY_COL, OUTCOME_PROBABILITY_COL
    )

    if len(df.index.unique()) != len(initial_pids):
        logger.warning(
            f"Number of unique PIDs reduced from {len(initial_pids)} to {len(df.index.unique())}"
        )

    df = _merge_with_predictions(
        df, counterfactual_predictions, OUTCOME_PROBABILITY_COL, TEMP_CF_COL
    )

    df = _assign_counterfactuals(df)
    df.drop(columns=[TEMP_CF_COL], inplace=True)

    logger.info(f"Final DataFrame shape: {df.shape}, Unique PIDs: {df.index.nunique()}")

    return df


def _merge_with_predictions(
    df: pd.DataFrame, predictions: pd.DataFrame, predictions_col: str, new_col_name: str
) -> pd.DataFrame:
    """Merge a DataFrame with predictions on their indices.

    Args:
        df: Input DataFrame to merge predictions into
        predictions: DataFrame containing the predictions to merge
        predictions_col: Name of column in predictions DataFrame containing the prediction values
        new_col_name: New name to give the predictions column in the merged DataFrame

    Returns:
        pd.DataFrame: DataFrame with predictions merged in under new_col_name

    Note:
        - Performs an inner merge on index, only keeping rows present in both DataFrames
        - Renames the predictions column to new_col_name before merging
        - Only merges the predictions column, discarding any other columns in predictions DataFrame
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
    treated_mask = df[TREATMENT_COL] == 1
    untreated_mask = ~treated_mask

    df[CF_TREATED_COL] = np.where(
        treated_mask, df[OUTCOME_PROBABILITY_COL], df[TEMP_CF_COL]
    )
    df[CF_CONTROL_COL] = np.where(
        untreated_mask, df[OUTCOME_PROBABILITY_COL], df[TEMP_CF_COL]
    )
    return df
