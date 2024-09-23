import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV


def get_indicator(t: int, T_observed: int, Y: bool, model_type: str) -> int:
    """Determine the event or censoring indicator based on model type."""
    if model_type == "failure":
        return 1 if ((t == T_observed) and (Y == 1)) else 0
    elif model_type == "censoring":
        return 1 if ((t == T_observed) and (Y == 0)) else 0
    else:
        raise ValueError("model_type must be either 'failure' or 'censoring'")


def transform_data_for_model_estimation(data: pd.DataFrame) -> pd.DataFrame:
    """Transform the data for either the failure model or the censoring model estimation.

    Args:
        data (pd.DataFrame): The input data containing T_observed, Y, A, X.
        model_type (str): Either 'failure' or 'censoring' to specify which model to transform the data for.

    Returns:
        pd.DataFrame: The transformed data suitable for classification.
    """
    expanded_rows = []

    # Loop over each patient in the original data
    for index, row in data.iterrows():
        T = int(row["T_observed"])
        expanded_rows.extend(
            [
                {
                    "pid": index,
                    "t": t,
                    "A": row["A"],
                    "X": row["X"],
                    "Y_E": get_indicator(t, T, row["Y"], "failure"),
                    "Y_C": get_indicator(t, T, row["Y"], "censoring"),
                }
                for t in range(0, T + 1)
            ]
        )

    # Create DataFrame from the list of rows
    expanded_data = pd.DataFrame(expanded_rows)
    return expanded_data


def estimate_survival_probability_at_T(data: pd.DataFrame, model: object) -> np.ndarray:
    """
    Estimate the survival probability at the final time for each patient for a given model (censoring or event).
    Args:
        data (pd.DataFrame): The input data containing T_observed, Y, A, X.
        model (object): The fitted censoring/failure model.

    Returns:
        np.ndarray: The estimated survival probabilities for each patient.
    """
    # Prepare the input features for prediction
    X = data[["t", "A", "X"]]

    # Predict the failure probabilities
    predicted_failure_probas = model.predict_proba(X)[:, 1]

    # Create a new DataFrame for predictions to avoid modifying the original data
    predictions = pd.DataFrame(
        {"pid": data["pid"], "predicted_failure_proba": predicted_failure_probas}
    )

    # Group by 'pid' and compute the survival probability
    # survival_probs = predictions.groupby('pid')['predicted_failure_proba'].apply(lambda x: np.prod(1 - x))
    survival_probs = (
        (1 - predictions.groupby("pid")["predicted_failure_proba"].transform("cumprod"))
        .groupby(predictions["pid"])
        .last()
    )
    return survival_probs.values


def select_pos_patients_at_risk(cls_data: pd.DataFrame, t_prime: int) -> pd.DataFrame:
    event_mask = cls_data["Y_E"] == 1
    event_data = cls_data[event_mask]
    pos_pids_at_risk = select_events_after_t_prime(event_data, t_prime)[
        "pid"
    ].unique()  # patients with an event at t_prime or later
    return select_patients(cls_data, pos_pids_at_risk)


def select_events_after_t_prime(cls_data: pd.DataFrame, t_prime: int) -> pd.DataFrame:
    return cls_data[cls_data["t"] > t_prime]


def select_patients(data: pd.DataFrame, pids: list) -> pd.DataFrame:
    return data[data["pid"].isin(pids)]


def select_treated(cls_data: pd.DataFrame) -> pd.DataFrame:
    """Select treated patients."""
    return cls_data[cls_data["A"] == 1]


def select_events_up_to_t_prime(cls_data: pd.DataFrame, t_prime: int) -> pd.DataFrame:
    """Select events up to time t_prime."""
    return cls_data[cls_data["t"] < t_prime]


def compute_IPCW_curve_at_t(
    N_patients: int, ps: np.ndarray, survival_probs: np.ndarray
) -> float:
    sum_ = np.sum(1 / (ps * survival_probs))
    return 1 / N_patients * sum_


# from time import time
def IPCW_estimator(
    cls_data: pd.DataFrame, data: pd.DataFrame, censoring_model: object
) -> np.ndarray:
    """
    Here we estimate the survival curve for treated patients using the IPCW estimator.
    For each time t, we select patients with outcome but still at risk at time t.
    We then compute for those the censoring survival probability at the time of outcome and the propensity score.
    """
    survival_curve = []
    max_time = cls_data["t"].max()
    censoring_survival_probs = estimate_survival_probability_at_T(
        cls_data, censoring_model
    )

    for t_prime in range(1, max_time):
        # 1. select treated patients with outcome but still at risk at time t
        cls_data_t_prime = select_pos_patients_at_risk(cls_data, t_prime)
        cls_data_t_prime = select_treated(cls_data_t_prime)
        # 2. Take only event up to time t_prime
        cls_data_t_prime = select_events_up_to_t_prime(cls_data_t_prime, t_prime)
        selected_ps = data.iloc[cls_data_t_prime["pid"].unique()]["propensity"]
        selected_survival_probs = censoring_survival_probs[
            cls_data_t_prime["pid"].unique()
        ]
        # 3. Compute the IPCW estimator
        # ! check what n needs to be
        n = cls_data[cls_data["A"] == 1]["pid"].nunique()
        survival_curve_at_t_prime = compute_IPCW_curve_at_t(
            n, selected_ps, selected_survival_probs
        )
        assert len(selected_ps) == len(selected_survival_probs), "Length mismatch"
        survival_curve.append(survival_curve_at_t_prime)
    return np.array(survival_curve), max_time


def estimate_ps(data: pd.DataFrame, model: object = LogisticRegressionCV(cv=5)):
    X = data["X"].values.reshape(-1, 1)
    treatment_model = model.fit(X, data["A"])
    data["propensity"] = treatment_model.predict_proba(X)[:, 1]
    return data


def fit_failure_model(
    cls_data: pd.DataFrame, model: object = GradientBoostingClassifier()
):
    """Fit the failure model."""
    return _fit_temporal_classifier(cls_data, "Y_E", model)


def fit_censoring_model(
    cls_data: pd.DataFrame, model: object = GradientBoostingClassifier()
):
    """Fit the censoring model."""
    return _fit_temporal_classifier(cls_data, "Y_C", model)


def _fit_temporal_classifier(
    cls_data: pd.DataFrame, target: str, model: object = GradientBoostingClassifier()
):
    """Fit a classifier with the target as the label.
    Target should be one of 'Y_E' or 'Y_C'.
    """
    if target not in ["Y_E", "Y_C"]:
        raise ValueError("target should be either 'Y_E' or 'Y_C'")
    X = cls_data[["t", "A", "X"]]
    Y = cls_data[target]
    return model.fit(X, Y)


def full_IPCW(data: pd.DataFrame) -> tuple:
    # 1. Fit the treatment model
    treatment_model = LogisticRegressionCV(cv=5)
    data = estimate_ps(data, treatment_model)

    # 2. Transform data to fit failure model
    cls_data = transform_data_for_model_estimation(data)

    # Step 3: Fit the censoring model
    censoring_model = fit_censoring_model(cls_data, GradientBoostingClassifier())

    ipcw_survival_curve, max_time = IPCW_estimator(cls_data, data, censoring_model)
    return ipcw_survival_curve, max_time
