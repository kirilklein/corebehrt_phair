# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: patbert
#     language: python
#     name: python3
# ---

# %%
import sys

if ".." not in sys.path:
    sys.path.append("..")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utils.estimates import estimate_causal_effects_with_multiple_methods
from utils.plot_sim import (
    plot_causal_effect_estimation_comparison,
    plot_propensity_score_dist_for_models,
    plot_calibration_curves,
)

from CausalEstimate.simulation.binary_simulation import (
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
    simulate_binary_data,
)
from utils.estimates import compute_ATE_from_data, compute_ATT_from_data


# %% [markdown]
# ## Define Models
#
#

# %%
alpha_base = [0, 0.2, -0.2]
beta_base = [-1, 2, 1, -1]

models = {
    "linear": {"alpha": alpha_base, "beta": beta_base},
    "treatment covariate interaction": {"alpha": alpha_base + [1], "beta": beta_base},
    "outcome covariate interaction": {"alpha": alpha_base, "beta": beta_base + [1]},
    "treatment and outcome covariate interaction": {
        "alpha": alpha_base + [1],
        "beta": beta_base + [1],
    },
    "nonlinear treatment effect": {
        "alpha": alpha_base,
        "beta": beta_base + [0, 0, 0, 1],
    },
    "extremely nonlinear": {
        "alpha": alpha_base + [1, -1, 1, -1],
        "beta": beta_base + [1, -1, 1, -1],
    },
    "semisynthetic mock": {
        "alpha": [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        "beta": [0, 2, 0, 0],
    },  # extremely nonlinear treatment assignment but outcome only linearly dependent on A
    "semisynthetic mock": {
        "alpha": [0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
        "beta": [0, 4, 0, 0],
    },  # same but stonger treatment effect
}
common_support_threshold = 0.05

# %% [markdown]
# ## Compare the two methods of computing the true effects

# %%
# Compare theoretical vs data-based computation of ATE and ATT
n_samples = 1000
seed = 42

ate_comparison = {}
att_comparison = {}

for model_name, params in models.items():
    data = simulate_binary_data(n_samples, params["alpha"], params["beta"], seed=seed)

    # Compute effects using both methods
    ate_comparison[model_name] = {
        "theoretical": compute_ATE_theoretical_from_data(data, params["beta"]),
        "data": compute_ATE_from_data(data),
    }
    ate_comparison[model_name]["difference"] = abs(
        ate_comparison[model_name]["theoretical"] - ate_comparison[model_name]["data"]
    )

    att_comparison[model_name] = {
        "theoretical": compute_ATT_theoretical_from_data(data, params["beta"]),
        "data": compute_ATT_from_data(data),
    }
    att_comparison[model_name]["difference"] = abs(
        att_comparison[model_name]["theoretical"] - att_comparison[model_name]["data"]
    )

# Print results
for effect_type, results in [("ATE", ate_comparison), ("ATT", att_comparison)]:
    print(f"\n{effect_type} Comparison:")
    print("-" * 50)
    for model, vals in results.items():
        print(f"\n{model}:")
        print(f"  Theoretical: {vals['theoretical']:.4f}")
        print(f"  Data-based:  {vals['data']:.4f}")
        print(f"  Difference:  {vals['difference']:.4f}")


# %% [markdown]
# ## Assess PS dists
#

# %%
fix, ax, dfs = plot_propensity_score_dist_for_models(
    models,
    ps_model=RandomForestClassifier,
    ps_model_kwargs={"n_estimators": 100, "max_depth": 5, "random_state": 42},
    calibration=True,
)

# %%
plot_calibration_curves(dfs)

# %% [markdown]
# ## Estimate Effects

# %%
methods = ["TMLE", "AIPW", "IPW"]
true_ate, estimated_ate, std_ate = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    n_samples=2000,
    effect_type="ATE",
    common_support_threshold=common_support_threshold,
)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
plot_causal_effect_estimation_comparison(
    ax,
    true_ate,
    estimated_ate,
    std_ate,
    methods,
    "Estimation Error Compared to True ATE: Logistic Regression",
    "ATE",
)
ax.set_ylim(-0.1, 0.12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RF classifier
#
#

# %% [markdown]
# ### ATE

# %%
rf_dic = {
    "model": RandomForestClassifier,
    "kwargs": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    "calibrate": True,
}

methods = [
    "TMLE",
    "AIPW",
    "IPW",
]
att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    n_samples=2000,
    effect_type="ATE",
    common_support_threshold=common_support_threshold,
    ps_model_dict=rf_dic,
    outcome_model_dict=rf_dic,
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATE - RF",
    "ATE",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ATT
#

# %%
methods = ["AIPW", "IPW"]

att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    n_samples=2000,
    common_support_threshold=common_support_threshold,
    ps_model_dict=rf_dic,
    outcome_model_dict=rf_dic,
)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATT - Logistic Regression",
    "ATT",
)
plt.tight_layout()
plt.show()
