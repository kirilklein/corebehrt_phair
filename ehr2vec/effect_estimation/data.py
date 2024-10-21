import logging

import numpy as np
import pandas as pd

from ehr2vec.common.default_args import (
    COUNTERFACTUAL_CONTROL_COL,
    COUNTERFACTUAL_TREATED_COL,
    OUTCOME_PREDICTIONS_COL,
    TREATMENT_COL,
)
from ehr2vec.data.utils import remove_duplicate_indices

logger = logging.getLogger(__name__)


def construct_data_for_effect_estimation(
    propensity_scores: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_predictions: pd.DataFrame = None,
    counterfactual_predictions: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Constructs the data for effect estimation from the propensity scores and outcomes dataframes.
    Returns a DataFrame with PID as index and the columns:
    proba (propensity scores), treatment status, and binary outcome.
    """
    # Perform an outer merge but only keep PIDs in propensities
    df = pd.merge(
        propensity_scores, outcomes, left_index=True, right_index=True, how="left"
    )
    df["outcome"].fillna(0, inplace=True)
    df["outcome"] = df["outcome"].astype(int)
    if counterfactual_predictions is not None and outcome_predictions is not None:
        df = add_outcome_predictions(
            df, counterfactual_predictions, outcome_predictions
        )

    return df


def add_outcome_predictions(
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

    df = merge_with_predictions(
        df, outcome_predictions, OUTCOME_PREDICTIONS_COL, OUTCOME_PREDICTIONS_COL
    )

    if len(df.index.unique()) != len(initial_pids):
        logger.warning(
            f"Number of unique PIDs reduced from {len(initial_pids)} to {len(df.index.unique())}"
        )

    df = merge_with_predictions(
        df, counterfactual_predictions, OUTCOME_PREDICTIONS_COL, "Y_hat_counterfactual"
    )

    df = assign_counterfactuals(df)
    df.drop(columns=["Y_hat_counterfactual"], inplace=True)

    logger.info(f"Final DataFrame shape: {df.shape}, Unique PIDs: {df.index.nunique()}")

    return df


def merge_with_predictions(
    df: pd.DataFrame, predictions: pd.DataFrame, predictions_col: str, new_col_name: str
) -> pd.DataFrame:
    """Merge df with predictions DataFrame on index."""
    predictions = predictions.rename(columns={predictions_col: new_col_name})
    return df.merge(
        predictions[[new_col_name]], left_index=True, right_index=True, how="inner"
    )


def assign_counterfactuals(df: pd.DataFrame) -> pd.DataFrame:
    """Assign Y1_hat and Y0_hat based on treatment status."""
    treated_mask = df[TREATMENT_COL] == 1
    untreated_mask = ~treated_mask

    df[COUNTERFACTUAL_TREATED_COL] = np.where(
        treated_mask, df[OUTCOME_PREDICTIONS_COL], df["Y_hat_counterfactual"]
    )
    df[COUNTERFACTUAL_CONTROL_COL] = np.where(
        untreated_mask, df[OUTCOME_PREDICTIONS_COL], df["Y_hat_counterfactual"]
    )
    return df


def construct_data_to_estimate_effect_from_counterfactuals(
    propensity_scores: pd.DataFrame, counterfactual_outcomes: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the data for effect estimation from the propensity scores and counterfactual outcomes dataframes.
    Returns a DataFrame with additional columns for Y1 and Y0.
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
