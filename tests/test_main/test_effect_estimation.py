"""This script is used to test the estimate_causal_effect.py script with different data generation and estimation methods."""

import os
import shutil
import subprocess
from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from sklearn.ensemble import RandomForestClassifier
from tests.common.plot_sim import plot_causal_effect_estimation_comparison
from tests.common.predictions import treatment_and_outcome_predictions

MAX_DEVIATION = 0.1

ESTIMATION_METHODS = {"ATE": ["IPW", "AIPW", "TMLE"], "ATT": ["IPW", "AIPW"]}
N_SAMPLES = 10000
N_BOOTSTRAPS = 50

ALPHA_BASE = [0, 0.2, -0.2]
BETA_BASE = [-1, 2, 1, -1]
MODELS = {
    "linear": {"alpha": ALPHA_BASE, "beta": BETA_BASE},
    "treatment covariate interaction": {"alpha": ALPHA_BASE + [1], "beta": BETA_BASE},
    "outcome covariate interaction": {"alpha": ALPHA_BASE, "beta": BETA_BASE + [1]},
    "treatment and outcome covariate interaction": {
        "alpha": ALPHA_BASE + [1],
        "beta": BETA_BASE + [1],
    },
    "nonlinear treatment effect": {
        "alpha": ALPHA_BASE,
        "beta": BETA_BASE + [0, 0, 0, 1],
    },
    "extremely nonlinear": {
        "alpha": ALPHA_BASE + [1, -1, 1, -1],
        "beta": BETA_BASE + [1, -1, 1, -1],
    },
    "azure exp. mock": {
        "alpha": [0, -0.1, 0.1, -0.5, 0.5, -0.5],
        "beta": [0, 4, 0, 0],
    },  # extremely nonlinear treatment assignment but outcome only linearly dependent on A
}


def generate_test_data(params, save_dir, n_samples=N_SAMPLES, seed=41):
    """Generate synthetic data for testing effect estimation."""
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


if __name__ == "__main__":
    # get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = join(script_dir, "tmp")

    true_effect_dict = defaultdict(dict)
    estimated_effect_dict = defaultdict(dict)
    std_effect_dict = defaultdict(dict)

    for i, (model, params) in enumerate(MODELS.items()):
        os.makedirs(save_dir, exist_ok=True)
        generate_test_data(params, save_dir)
        # Create minimal config file
        for effect_type, methods in ESTIMATION_METHODS.items():
            config = {
                "env": "local",
                "paths": {
                    "run_name": "test_run",
                    "output_path": f"{save_dir}/test_output",
                    "ps_model_path": f"{save_dir}/test_data",
                    "outcome": f"{save_dir}/test_data/outcomes.csv",
                    "counterfactual_outcome": f"{save_dir}/test_data/counterfactual_outcomes.csv",
                    "outcome_predictions": f"{save_dir}/test_data/outcome_predictions.csv",
                    "outcome_predictions_counterfactual": f"{save_dir}/test_data/counterfactual_predictions.csv",
                },
                "ps_file": "ps_scores.csv",
                "estimator": {
                    "methods": methods,
                    "effect_type": effect_type,
                    "n_bootstrap": N_BOOTSTRAPS,
                    "common_support_threshold": 0.05,
                },
            }

            # Save config
            with open(join(save_dir, "test_config.yaml"), "w") as f:
                yaml.dump(config, f)

            # # Run the effect estimation script
            subprocess.run(
                [
                    "python",
                    "ehr2vec/main/06_estimate_causal_effect.py",
                    "--config_path",
                    join(save_dir, "test_config.yaml"),
                ],
                check=True,
            )

            estimated_effects = pd.read_csv(
                join(save_dir, "test_output/experiment_test_run/effect.csv")
            )
            true_effect = estimated_effects["effect_counterfactual"].iloc[0]
            for _, row in estimated_effects.iterrows():
                print(
                    f"Method: {row['method']}, Estimate: {row['effect']:.3f}, True {effect_type}: {true_effect:.3f}, Difference: {(row['effect'] - true_effect):.3f}"
                )
                if abs(true_effect - row["effect"]) > MAX_DEVIATION:
                    raise ValueError(
                        f"True effect {true_effect} is outside of the acceptable range for {effect_type} {model} {row['method']}"
                    )
            # clean up

            estimated_effect_dict[effect_type][model] = {
                row["method"]: row["effect"] for _, row in estimated_effects.iterrows()
            }
            std_effect_dict[effect_type][model] = {
                row["method"]: row["std_err"] for _, row in estimated_effects.iterrows()
            }
            true_effect_dict[effect_type][model] = true_effect

            shutil.rmtree(f"{save_dir}/test_output")
        shutil.rmtree(save_dir)

    for effect_type in ESTIMATION_METHODS.keys():
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_causal_effect_estimation_comparison(
            ax,
            true_effect_dict[effect_type],
            estimated_effect_dict[effect_type],
            std_effect_dict[effect_type],
            ESTIMATION_METHODS[effect_type],
            "Estimation Error Compared to True Effect",
            effect_type,
        )
        plt.tight_layout()
        plt.show()
        os.makedirs(f"{script_dir}/outputs", exist_ok=True)
        fig.savefig(
            f"{script_dir}/outputs/effect_estimation_script_test_results_{effect_type}.png"
        )
