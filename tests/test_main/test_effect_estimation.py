import os
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from CausalEstimate.filter.propensity import filter_common_support
from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../ehr2vec/")

from notebooks.utils.plot_sim import plot_causal_effect_estimation_comparison
from notebooks.utils.predictions import treatment_and_outcome_predictions

METHODS = ["IPW", "AIPW", "TMLE"]
EFFECT_TYPE = "ATE"
N_SAMPLES = 10000
N_BOOTSTRAPS = 50

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

def generate_test_data(params,n_samples=N_SAMPLES, seed=41):
    """Generate synthetic data for testing effect estimation."""
    # Simulate data
    data = simulate_binary_data(n_samples, alpha=params["alpha"], beta=params["beta"], seed=seed)
    data["pid"] = range(n_samples)
    
    # Configure models
    model_config = {
        "model": RandomForestClassifier,
        "kwargs": {"n_estimators": 100, "max_depth": 5, "random_state": seed},
        "calibrate": True
    }
    # Generate predictions
    data_with_preds = treatment_and_outcome_predictions(
        data,
        ps_model_dict=model_config,
        outcome_model_dict=model_config,
        n_splits=5
    )

    # Combine Q1 and Q0 into counterfactual predictions based on treatment status
    data_with_preds['Q*'] = np.where(data_with_preds['A'] == 1, data_with_preds['Q0'], data_with_preds['Q1'])

    data_with_preds['Y0'] = np.where(data_with_preds['A'] == 0, data_with_preds['Y'], data_with_preds['Y_cf'])
    data_with_preds['Y1'] = np.where(data_with_preds['A'] == 1, data_with_preds['Y'], data_with_preds['Y_cf'])
    
    # Prepare output files
    ps_df = pd.DataFrame({
        'pid': data_with_preds['pid'],
        'target': data_with_preds['A'],
        'proba': data_with_preds['ps']
    })
    
    # Create outcomes dataframe with only positive cases and random timestamps
    outcomes_df = pd.DataFrame({
        'PID': data_with_preds['pid'],
        'TIMESTAMP': pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    })
    outcomes_df = outcomes_df[data_with_preds['Y'] == 1].reset_index(drop=True)
    
    # Create counterfactual outcomes dataframe with true Y0/Y1 values
    counterfactual_outcomes_df = pd.DataFrame({
        'PID': data_with_preds['pid'],
        'Y0': data_with_preds['Y0'],
        'Y1': data_with_preds['Y1']
    })

    outcome_predictions_df = pd.DataFrame({
        'pid': data_with_preds['pid'],
        'proba': data_with_preds['Q']
    })
    
    counterfactual_predictions_df = pd.DataFrame({
        'pid': data_with_preds['pid'],
        'proba': data_with_preds['Q*']
    })
    
    # Save files
    output_dir = "tmp/test_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving files to {output_dir}")
    ps_df.to_csv(f"{output_dir}/ps_scores.csv", index=False)
    outcomes_df.to_csv(f"{output_dir}/outcomes.csv", index=False)
    counterfactual_outcomes_df.to_csv(f"{output_dir}/counterfactual_outcomes.csv", index=False)
    outcome_predictions_df.to_csv(f"{output_dir}/outcome_predictions.csv", index=False)
    counterfactual_predictions_df.to_csv(f"{output_dir}/counterfactual_predictions.csv", index=False)

    # Filter common support
    filtered_data = filter_common_support(data_with_preds, ps_col='ps', treatment_col='A', threshold=0.05)
    true_ate = compute_ATE_from_data(filtered_data)
    print("True ATE: ", true_ate)
    # return true_ate 
    
if __name__ == "__main__":
    # Generate test data
    alpha_base = [0, 0.2, -0.2]
    beta_base = [-1, 2, 1, -1]
    models = {
        "linear": {"alpha": alpha_base, "beta": beta_base},
        "treatment covariate interaction": {"alpha": alpha_base + [1], "beta": beta_base},
        "outcome covariate interaction": {"alpha": alpha_base, "beta": beta_base + [1]},
        "treatment and outcome covariate interaction": {"alpha": alpha_base + [1], "beta": beta_base + [1]},
        "nonlinear treatment effect": {"alpha": alpha_base, "beta": beta_base + [0, 0, 0, 1]},
        "extremely nonlinear": {"alpha": alpha_base + [1, -1, 1, -1], "beta": beta_base + [1, -1, 1, -1]},
        "azure exp. mock": {"alpha": [0, -.1, .1, -.5, .5, -.5], "beta": [0, 4, 0, 0]}, # extremely nonlinear treatment assignment but outcome only linearly dependent on A
    }
    true_effect_dict = {}
    estimated_effect_dict = {}
    std_effect_dict = {}
    for i, (model, params) in enumerate(models.items()):
        if i>1:
            break
        generate_test_data(params)
        
        # Create minimal config file
        config = {
            "env": "local",
            "paths": {
                "run_name": "test_run",
                "output_path": "tmp/test_output",
                "ps_model_path": "tmp/test_data",
                "outcome": "tmp/test_data/outcomes.csv",
                "counterfactual_outcome": "tmp/test_data/counterfactual_outcomes.csv",
                "outcome_predictions": "tmp/test_data/outcome_predictions.csv",
                "outcome_predictions_counterfactual": "tmp/test_data/counterfactual_predictions.csv",
            },
            "ps_file": "ps_scores.csv",
            "estimator": {
                "methods": METHODS,
                "effect_type": EFFECT_TYPE,
                "n_bootstrap": N_BOOTSTRAPS,
                "common_support_threshold": 0.05
            }
        }
        
        # Save config
        with open("tmp/test_config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # # Run the effect estimation script
        subprocess.run(["python", "ehr2vec/main/06_estimate_causal_effect.py", "--config_path", "../../tmp/test_config.yaml"], check=True)

        estimated_effects = pd.read_csv("tmp/test_output/experiment_test_run/effect.csv")
        true_effect = estimated_effects["effect_counterfactual"].iloc[0]
        for _, row in estimated_effects.iterrows():
            print(f"Method: {row['method']}, Estimate: {row['effect']:.3f}, True {EFFECT_TYPE}: {true_effect:.3f}, Difference: {(row['effect'] - true_effect):.3f}")
        # clean up

        estimated_effect_dict[model] = {
            row["method"]: row["effect"] for _, row in estimated_effects.iterrows()
        }
        std_effect_dict[model] = {
            row["method"]: row["std_err"] for _, row in estimated_effects.iterrows()
        }
        true_effect_dict[model] = true_effect
        shutil.rmtree("tmp/test_data")
        shutil.rmtree("tmp/test_output")
        os.remove("tmp/test_config.yaml")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_causal_effect_estimation_comparison(ax, true_effect_dict, estimated_effect_dict, std_effect_dict, METHODS, "Estimation Error Compared to True Effect", EFFECT_TYPE)
    plt.tight_layout()
    plt.show()
    fig.savefig("tmp/test_results.png")

