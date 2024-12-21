import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tests.common.estimates import treatment_and_outcome_predictions
from CausalEstimate.simulation.binary_simulation import simulate_binary_data


def generate_test_data_with_predictions(
    params: dict, save_dir: str, n_samples: int = 10000, seed: int = 41
) -> None:
    """
    Generate synthetic data and predictions for testing effect estimation.
    args:
        params: parameters for the simulation
        save_dir: directory to save the data
        n_samples: number of samples to generate
        seed: random seed
    Saves:
        ps_df: propensity scores
        outcomes_df: outcomes
        counterfactual_outcomes_df: counterfactual outcomes
        outcome_predictions_df: outcome predictions
        counterfactual_predictions_df: counterfactual predictions
    """

    save_dir = join(save_dir, "test_data")
    os.makedirs(save_dir, exist_ok=True)
    # Simulate data
    data = simulate_binary_data(
        n_samples, alpha=params["alpha"], beta=params["beta"], seed=seed
    )
    data["pid"] = range(n_samples)

    # Configure models
    model_config = {
        "model": RandomForestClassifier,
        "kwargs": {"n_estimators": 100, "max_depth": 5, "random_state": seed},
        "calibrate": True,
    }
    # Generate predictions
    data_with_preds = treatment_and_outcome_predictions(
        data, ps_model_dict=model_config, outcome_model_dict=model_config, n_splits=5
    )

    # Combine Q1 and Q0 into counterfactual predictions based on treatment status
    data_with_preds["Q*"] = np.where(
        data_with_preds["A"] == 1, data_with_preds["Q0"], data_with_preds["Q1"]
    )

    data_with_preds["Y0"] = np.where(
        data_with_preds["A"] == 0, data_with_preds["Y"], data_with_preds["Y_cf"]
    )
    data_with_preds["Y1"] = np.where(
        data_with_preds["A"] == 1, data_with_preds["Y"], data_with_preds["Y_cf"]
    )

    # Prepare output files
    ps_df = pd.DataFrame(
        {
            "pid": data_with_preds["pid"],
            "target": data_with_preds["A"],
            "proba": data_with_preds["ps"],
        }
    )

    # Create outcomes dataframe with only positive cases and random timestamps
    outcomes_df = pd.DataFrame(
        {
            "PID": data_with_preds["pid"],
            "TIMESTAMP": pd.date_range(start="2020-01-01", periods=n_samples, freq="D"),
        }
    )
    outcomes_df = outcomes_df[data_with_preds["Y"] == 1].reset_index(drop=True)

    # Create counterfactual outcomes dataframe with true Y0/Y1 values
    counterfactual_outcomes_df = pd.DataFrame(
        {
            "PID": data_with_preds["pid"],
            "Y0": data_with_preds["Y0"],
            "Y1": data_with_preds["Y1"],
        }
    )

    outcome_predictions_df = pd.DataFrame(
        {"pid": data_with_preds["pid"], "proba": data_with_preds["Q"]}
    )

    counterfactual_predictions_df = pd.DataFrame(
        {"pid": data_with_preds["pid"], "proba": data_with_preds["Q*"]}
    )

    # Save files
    ps_df.to_csv(f"{save_dir}/ps_scores.csv", index=False)
    outcomes_df.to_csv(f"{save_dir}/outcomes.csv", index=False)
    counterfactual_outcomes_df.to_csv(
        f"{save_dir}/counterfactual_outcomes.csv", index=False
    )
    outcome_predictions_df.to_csv(f"{save_dir}/outcome_predictions.csv", index=False)
    counterfactual_predictions_df.to_csv(
        f"{save_dir}/counterfactual_predictions.csv", index=False
    )
