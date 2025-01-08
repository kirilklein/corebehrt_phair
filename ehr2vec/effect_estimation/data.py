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
    Constructs the data for effect estimation from propensity scores and outcomes.
    Returns a DataFrame with PID as index and the required columns.
    """
    df = merge_dataframes(
        propensity_scores, outcomes, fillna_col="outcome", fill_value=0
    )
    df["outcome"] = df["outcome"].astype(int)

    if outcome_predictions is not None and counterfactual_predictions is not None:
        df = process_predictions(df, outcome_predictions, counterfactual_predictions)

    return df


def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    fillna_col: str = None,
    fill_value=None,
) -> pd.DataFrame:
    """Merges two DataFrames on their indices, optionally filling missing values."""
    merged_df = pd.merge(
        left_df, right_df, left_index=True, right_index=True, how="left"
    )
    if fillna_col and fill_value is not None:
        merged_df[fillna_col] = merged_df[fillna_col].fillna(fill_value)
    return merged_df


def process_predictions(
    df: pd.DataFrame,
    outcome_predictions: pd.DataFrame,
    counterfactual_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Processes and integrates outcome and counterfactual predictions into the DataFrame.
    """
    df = remove_duplicate_indices(df)
    outcome_predictions = remove_duplicate_indices(outcome_predictions)
    counterfactual_predictions = remove_duplicate_indices(counterfactual_predictions)

    initial_pids = df.index.nunique()
    df = merge_with_predictions(
        df, outcome_predictions, OUTCOME_PREDICTIONS_COL, OUTCOME_PREDICTIONS_COL
    )

    if df.index.nunique() != initial_pids:
        logger.warning(
            f"Unique PIDs reduced from {initial_pids} to {df.index.nunique()} during merge with predictions."
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
    """Merges the DataFrame with predictions, renaming the specified column."""
    predictions = predictions.rename(columns={predictions_col: new_col_name})
    return df.merge(
        predictions[[new_col_name]], left_index=True, right_index=True, how="inner"
    )


def assign_counterfactuals(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns Y1_hat and Y0_hat columns based on treatment status."""
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
    Constructs data for effect estimation from propensity scores and counterfactual outcomes.
    """
    counterfactual_outcomes = counterfactual_outcomes.set_index("PID")
    return pd.merge(
        propensity_scores,
        counterfactual_outcomes,
        left_index=True,
        right_index=True,
        how="left",
        validate="one_to_one",
    )
