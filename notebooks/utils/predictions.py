from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def treatment_and_outcome_predictions(
    data,
    ps_model=BaseEstimator,
    outcome_model=BaseEstimator,
    ps_model_kwargs={},
    outcome_model_kwargs={},
    calibrate=True,
):
    data["ps"] = predict_propensity_cv(
        data,
        ["X1", "X2"],
        "A",
        n_splits=5,
        model=ps_model,
        ps_model_kwargs=ps_model_kwargs,
        calibrate=calibrate,
    )
    data["Q"], data["Q*"] = predict_outcome_cv(
        data,
        ["X1", "X2"],
        "Y",
        "A",
        n_splits=5,
        model=outcome_model,
        outcome_model_kwargs=outcome_model_kwargs,
        calibrate=calibrate,
    )
    data["Q1"] = np.where(
        data["A"] == 1, data["Q"], data["Q*"]
    )  # If A=1, Q is already the scenario of A=1; else use Q*
    data["Q0"] = np.where(
        data["A"] == 0, data["Q"], data["Q*"]
    )  # If A=0, Q is scenario of A=0; else use Q*
    return data


def _fit_and_predict_calibrated(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    clf_model,
    model_kwargs,
    calibrate=True,
) -> Tuple[np.ndarray, BaseEstimator, BaseEstimator]:
    """Helper function to fit model with calibration and make predictions"""
    # Fit RF and calibrate
    clf = clf_model(**model_kwargs)
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]

    # Clip probabilities to avoid numerical instability
    train_probs = np.clip(train_probs, 1e-7, 1 - 1e-7)

    # Predict and clip validation probabilities
    val_probs = clf.predict_proba(X_val)[:, 1]

    calibrator = None
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(train_probs, y_train)
        val_probs = calibrator.transform(val_probs)
        val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)
    return val_probs, clf, calibrator


def predict_propensity_cv(
    data: pd.DataFrame,
    X_cols: list,
    treatment_col: str = "A",
    n_splits: int = 5,
    model: BaseEstimator = LogisticRegression,
    ps_model_kwargs: dict = {},
    calibrate: bool = True,
) -> np.ndarray:
    """Cross-validated propensity score predictions"""
    X = data[X_cols].copy()
    y = data[treatment_col].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = np.zeros(len(data))

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        predictions[val_idx], _, _ = _fit_and_predict_calibrated(
            X_train,
            X_val,
            y_train,
            model,
            model_kwargs=ps_model_kwargs,
            calibrate=calibrate,
        )

    return predictions


def predict_outcome_cv(
    data: pd.DataFrame,
    X_cols: list,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    n_splits: int = 5,
    model: BaseEstimator = LogisticRegression,
    outcome_model_kwargs: dict = {},
    calibrate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-validated outcome predictions for factual and counterfactual"""
    X = data[X_cols + [treatment_col]].copy()
    X_counter = X.copy()
    X_counter.loc[:, treatment_col] = 1 - X[treatment_col]

    y = data[outcome_col].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    factual_preds = np.zeros(len(data))
    counter_preds = np.zeros(len(data))

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        X_counter_val = X_counter.iloc[val_idx].copy()
        y_train = y.iloc[train_idx]

        X_train.loc[:, treatment_col] = data.iloc[train_idx][treatment_col]

        factual_preds[val_idx], clf, calibrator = _fit_and_predict_calibrated(
            X_train,
            X_val,
            y_train,
            model,
            model_kwargs=outcome_model_kwargs,
            calibrate=calibrate,
        )

        counter_preds[val_idx] = clf.predict_proba(X_counter_val)[:, 1]
        if calibrate:
            counter_preds[val_idx] = calibrator.transform(counter_preds[val_idx])

    return factual_preds, counter_preds
