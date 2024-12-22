from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.interface.estimator import Estimator
from CausalEstimate.simulation.binary_simulation import (
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
    simulate_binary_data,
)
from sklearn.linear_model import LogisticRegression
from tests.common.predictions import treatment_and_outcome_predictions


def estimate_causal_effects_with_multiple_methods(
    models: Dict[str, Dict[str, Any]],
    methods: List[str],
    effect_type: str = "ATE",
    n_samples: int = 3000,
    common_support_threshold: float = 0.05,
    n_bootstraps: int = 30,
    n_splits: int = 5,
    ps_model_dict: Dict = None,
    outcome_model_dict: Dict = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Estimate causal effects using multiple methods and compare to true effects.
    For each model specification, this function:
    1. Simulates binary treatment-outcome data
    2. Fits propensity score and outcome models
    3. Estimates causal effects using specified methods
    4. Computes true effects for comparison

    Args:
        models: Dict mapping model names to dicts containing alpha and beta parameters
        methods: List of causal estimation methods to use (e.g. ['IPW', 'AIPW'])
        effect_type: Type of causal effect to estimate ('ATE' or 'ATT')
        n_samples: Number of samples to simulate
        common_support_threshold: Threshold for common support filtering
        n_bootstraps: Number of bootstrap iterations for CI estimation
        n_splits: Number of cross-validation splits for model fitting
        ps_model_dict: Dict containing propensity score model config with keys:
            'model': Model class
            'kwargs': Model parameters
            'calibrate': Whether to calibrate predictions
        outcome_model_dict: Dict containing outcome model config with same structure as ps_model_dict

    Returns:
        true_effect: Dict mapping model names to true causal effects
        estimated_effect: Dict mapping model names to dicts of estimated effects by method
        std_effect: Dict mapping model names to dicts of effect standard errors by method
    """
    if ps_model_dict is None:
        ps_model_dict = {
            "model": LogisticRegression,
            "kwargs": {
                "penalty": None,
                "solver": "lbfgs",
                "max_iter": 1000,
            },
            "calibrate": True,
        }
    if outcome_model_dict is None:
        outcome_model_dict = {
            "model": LogisticRegression,
            "kwargs": {
                "penalty": None,
                "solver": "lbfgs",
                "max_iter": 1000,
            },
            "calibrate": True,
        }

    true_effect = {}
    estimated_effect = {}
    std_effect = {}
    for model, params in models.items():
        data = simulate_binary_data(
            n_samples, alpha=params["alpha"], beta=params["beta"], seed=44
        )
        data = treatment_and_outcome_predictions(
            data,
            ps_model_dict=ps_model_dict,
            outcome_model_dict=outcome_model_dict,
            n_splits=n_splits,
        )

        ate_estimate = Estimator(
            methods=methods, effect_type=effect_type
        ).compute_effect(
            data,
            treatment_col="A",
            outcome_col="Y",
            ps_col="ps",
            predicted_outcome_treated_col="Q1",
            predicted_outcome_control_col="Q0",
            predicted_outcome_col="Q",
            bootstrap=True,
            n_bootstraps=n_bootstraps,
            common_support_threshold=common_support_threshold,
        )

        data = filter_common_support(
            data, ps_col="ps", treatment_col="A", threshold=common_support_threshold
        )
        if effect_type == "ATE":
            true_effect[model] = compute_ATE_theoretical_from_data(data, params["beta"])
        elif effect_type == "ATT":
            true_effect[model] = compute_ATT_theoretical_from_data(data, params["beta"])

        estimated_effect[model] = {
            method: ate_estimate[method]["effect"] for method in methods
        }
        std_effect[model] = {
            method: ate_estimate[method]["std_err"] for method in methods
        }
    return true_effect, estimated_effect, std_effect


def compute_ATE_from_data(data: pd.DataFrame):
    """Compute the true average treatment effect (ATE) from the data using counterfactuals.

    Args:
        data: DataFrame containing Y (observed outcome), Y_cf (counterfactual outcome),
              and A (treatment assignment)
        beta: List of coefficients (unused, kept for API compatibility)

    Returns:
        float: Average treatment effect
    """
    # For treated (A=1): effect is Y - Y_cf
    # For control (A=0): effect is Y_cf - Y
    individual_effects = np.where(
        data["A"] == 1, data["Y"] - data["Y_cf"], data["Y_cf"] - data["Y"]
    )

    return individual_effects.mean()


def compute_ATT_from_data(data: pd.DataFrame):
    """Compute the true average treatment effect on the treated (ATT) from the data using counterfactuals."""
    treated_data = data[data["A"] == 1]
    individual_effects = treated_data["Y"] - treated_data["Y_cf"]
    return individual_effects.mean()


def compare_treatment_effect_estimates_with_ground_truth_across_patient_numbers(
    patient_numbers,
    models,
    estimators,
    n_bootstraps=100,
    effect_type="ATE",
    ps_model=LogisticRegression,
    outcome_model=LogisticRegression,
    ps_model_kwargs={},
    outcome_model_kwargs={},
    calibrate=True,
    common_support_threshold=0.01,
):
    """
    Get the difference between the true effect and the estimated effect for each model and patient number.

    Args:
        patient_numbers: List of patient numbers to simulate
        models: Dictionary of models with their parameters
        estimators: List of estimator methods to use
        n_bootstraps: Number of bootstrap iterations
        effect_type: Type of effect to estimate ('ATE' or 'ATT')
        ps_model: Propensity score model class
        outcome_model: Outcome model class
        ps_model_kwargs: Keyword arguments for propensity score model
        outcome_model_kwargs: Keyword arguments for outcome model
        calibrate: Whether to calibrate predictions
        common_support_threshold: Threshold for common support filtering

    Returns:
        diffs: Dictionary of differences between estimated and true effects
        stds: Dictionary of standard errors
    """
    diffs = defaultdict(lambda: defaultdict(list))
    stds = defaultdict(lambda: defaultdict(list))

    for n_samples in patient_numbers:
        print(f"Patient number: {n_samples}")
        true_effect, estimated_effect, std_effect = (
            estimate_causal_effects_with_multiple_methods(
                n_samples,  # Add missing n_samples parameter
                models,
                estimators,
                n_bootstraps=n_bootstraps,
                effect_type=effect_type,
                ps_model=ps_model,
                outcome_model=outcome_model,
                ps_model_kwargs=ps_model_kwargs,
                outcome_model_kwargs=outcome_model_kwargs,
                calibrate=calibrate,
                common_support_threshold=common_support_threshold,
            )
        )

        # Store differences and standard errors
        for est in estimators:
            for model_name in models:
                diffs[est][model_name].append(
                    estimated_effect[model_name][est] - true_effect[model_name]
                )
                stds[est][model_name].append(std_effect[model_name][est])

    # Convert lists to numpy arrays
    for estimator in estimators:
        for model_name in models:
            diffs[estimator][model_name] = np.array(diffs[estimator][model_name])
            stds[estimator][model_name] = np.array(stds[estimator][model_name])

    return diffs, stds
