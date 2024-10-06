import pandas as pd

def construct_data_for_effect_estimation(
    propensity_scores: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs the data for effect estimation from the propensity scores and outcomes dataframes.
    Returns a DataFrame with PID as index and the columns:
    proba (propensity scores), treatment status, and binary outcome.
    """
    # Perform an outer merge but only keep PIDs in propensities
    df = pd.merge(propensity_scores, outcomes, left_index=True, right_index=True, how="left")
    df["outcome"].fillna(0, inplace=True)
    df["outcome"] = df["outcome"].astype(int)
    return df