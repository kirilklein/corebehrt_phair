from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

CLIP = 1e-3


def treatment_and_outcome_predictions(
    data,
    ps_model_dict: dict = {"model": BaseEstimator, "kwargs": {}, "calibrate": True},
    outcome_model_dict: dict = {
        "model": BaseEstimator,
        "kwargs": {},
        "calibrate": True,
    },
    n_splits: int = 5,
):
    """Generate propensity score and outcome predictions using cross-validation.

    This function performs two main tasks:
    1. Estimates propensity scores (probability of treatment) using cross-validation
    2. Estimates potential outcomes under treatment and control using cross-validation

    Args:
        data: DataFrame containing covariates (X1, X2), treatment (A), and outcome (Y)
        ps_model_dict: Dictionary specifying propensity score model configuration with keys:
            - model: Model class to use (default: BaseEstimator)
            - kwargs: Model parameters (default: {})
            - calibrate: Whether to calibrate predictions (default: True)
        outcome_model_dict: Dictionary specifying outcome model configuration with same structure as ps_model_dict
        n_splits: Number of cross-validation folds (default: 5)

    Returns:
        DataFrame with added columns:
            - ps: Propensity scores
            - Q: Predicted outcomes under observed treatment
            - Q1: Predicted outcomes under treatment (A=1)
            - Q0: Predicted outcomes under control (A=0)
    """
    data["ps"] = predict_propensity_cv(
        data,
        X_cols=["X1", "X2"],
        treatment_col="A",
        n_splits=n_splits,
        model=ps_model_dict["model"],
        ps_model_kwargs=ps_model_dict["kwargs"],
        calibrate=ps_model_dict.get("calibrate", True),
    )
    data["Q"], data["Q*"] = predict_outcome_cv(
        data,
        X_cols=["X1", "X2"],
        outcome_col="Y",
        treatment_col="A",
        n_splits=n_splits,
        model=outcome_model_dict["model"],
        outcome_model_kwargs=outcome_model_dict["kwargs"],
        calibrate=outcome_model_dict.get("calibrate", True),
    )
    data["Q1"] = np.where(
        data["A"] == 1, data["Q"], data["Q*"]
    )  # If A=1, Q is already the scenario of A=1; else use Q*
    data["Q0"] = np.where(
        data["A"] == 0, data["Q"], data["Q*"]
    )  # If A=0, Q is scenario of A=0; else use Q*
    data.drop(columns=["Q*"], inplace=True)
    return data


def _fit_and_predict_calibrated(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    clf_model: BaseEstimator,
    model_kwargs: dict,
    calibrate: bool = True,
) -> Tuple[np.ndarray, BaseEstimator, BaseEstimator]:
    """Fits a classifier model with optional probability calibration and makes predictions.

    This helper function fits a classifier model on training data and optionally calibrates
    its probability predictions using isotonic regression. It then generates predictions
    on validation data.

    Args:
        X_train: Training feature matrix
        X_val: Validation feature matrix
        y_train: Training labels
        clf_model: Classifier model class to use (e.g. LogisticRegression)
        model_kwargs: Dictionary of keyword arguments for classifier initialization
        calibrate: Whether to calibrate probability predictions using IsotonicRegression

    Returns:
        Tuple containing:
            - val_probs: Calibrated probability predictions on validation data
            - clf: Fitted classifier model
            - calibrator: Fitted calibration model (None if calibrate=False)
    """
    # Fit RF and calibrate
    clf = clf_model(**model_kwargs)
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    # train_probs = _clip_predictions(train_probs)

    val_probs = clf.predict_proba(X_val)[:, 1]
    val_probs = _clip_predictions(val_probs)

    calibrator = None
    if calibrate:
        calibrator = IsotonicRegression(
            out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6
        )
        calibrator.fit(train_probs, y_train)
        val_probs = calibrator.predict(val_probs)
        val_probs = _clip_predictions(val_probs)
    return val_probs, clf, calibrator


def predict_propensity_cv(
    data: pd.DataFrame,
    X_cols: list,
    model: BaseEstimator,
    ps_model_kwargs: dict,
    treatment_col: str = "A",
    n_splits: int = 5,
    calibrate: bool = True,
) -> np.ndarray:
    """Generates cross-validated propensity score predictions using k-fold CV.

    This function performs k-fold cross validation to generate propensity score
    predictions for each sample. For each fold, it fits a classifier on the training
    data and generates predictions for the validation fold. The predictions can
    optionally be calibrated using isotonic regression.

    Args:
        data: DataFrame containing features and treatment assignment
        X_cols: List of feature column names to use for prediction
        treatment_col: Name of binary treatment assignment column
        n_splits: Number of cross-validation folds
        model: Classifier model class to use (e.g. LogisticRegression)
        ps_model_kwargs: Dictionary of keyword arguments for classifier initialization
        calibrate: Whether to calibrate probability predictions using IsotonicRegression

    Returns:
        Array of propensity score predictions for each sample, with values between 0 and 1
    """
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
    model: BaseEstimator,
    outcome_model_kwargs: dict,
    outcome_col: str = "Y",
    treatment_col: str = "A",
    n_splits: int = 5,
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
        counter_preds[val_idx] = _clip_predictions(counter_preds[val_idx])
        if calibrate:
            counter_preds[val_idx] = calibrator.transform(counter_preds[val_idx])
            counter_preds[val_idx] = _clip_predictions(counter_preds[val_idx])
    return factual_preds, counter_preds


def _clip_predictions(predictions: np.ndarray) -> np.ndarray:
    """Clip predictions to avoid numerical instability"""
    return np.clip(predictions, CLIP, 1 - CLIP)
