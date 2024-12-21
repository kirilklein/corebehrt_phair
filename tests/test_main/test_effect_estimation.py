"""This script is used to test the estimate_causal_effect.py script with different data generation and estimation methods."""

import os
import shutil
import subprocess
from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tests.common.generate import generate_test_data_with_predictions
from tests.common.plot_sim import plot_causal_effect_estimation_comparison

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


if __name__ == "__main__":
    # get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = join(script_dir, "tmp")

    true_effect_dict = defaultdict(dict)
    estimated_effect_dict = defaultdict(dict)
    std_effect_dict = defaultdict(dict)

    for i, (model, params) in enumerate(MODELS.items()):
        os.makedirs(save_dir, exist_ok=True)
        generate_test_data_with_predictions(
            params, save_dir, n_samples=N_SAMPLES, seed=i
        )
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
