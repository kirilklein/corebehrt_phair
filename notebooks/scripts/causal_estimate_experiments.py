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

from utils.estimates import estimate_causal_effects_with_multiple_methods

import matplotlib.pyplot as plt
import numpy as np


# %% [markdown]
# ## ATE
#


# %%
def plot_causal_effect_estimation_comparison(
    ax: plt.Axes,
    true_effect: dict,
    estimated_effect: dict,
    std_effect: dict,
    methods: list,
    title: str,
    effect_type: str,
):
    """Plot comparison of estimated vs true causal effects across different models and methods.

    Args:
        ax (plt.Axes): Matplotlib axes object to plot on
        true_effect (dict): Dictionary mapping model names to true causal effects
        estimated_effect (dict): Dictionary mapping model names to dictionaries of estimated effects by method
        std_effect (dict): Dictionary mapping model names to dictionaries of effect standard errors by method
        methods (list): List of causal estimation methods to plot
        title (str): Plot title
        effect_type (str): Type of causal effect being plotted ('ATE' or 'ATT')

    Returns:
        plt.Axes: The axes object with the plotted results

    The plot shows bars representing the difference between estimated and true effects for each
    model and method combination. Error bars show standard errors of the estimates. True effect
    values are displayed as text. A horizontal line at 0 indicates perfect estimation.
    """
    diffs_all = {}
    models = estimated_effect.keys()
    for model in models:
        diffs_all[model] = {
            method: estimated_effect[model][method] - true_effect[model]
            for method in methods
        }
    x = np.arange(len(models))
    width = 0.2  # Reduced width for less overlap
    colors = [
        "#2ecc71",
        "#3498db",
        "#e74c3c",
    ]  # Emerald green, Dodger blue, Alizarin red
    for i, method in enumerate(methods):
        diffs = [diffs_all[model][method] for model in models]
        stds = [std_effect[model][method] for model in models]
        ax.bar(
            x + (i - 1) * width, diffs, width, label=method, color=colors[i], alpha=0.8
        )
        # Add error bars using the standard deviations
        ax.errorbar(
            x + (i - 1) * width,
            diffs,
            yerr=stds,
            fmt="none",
            color="black",
            capsize=3,
            alpha=0.5,
        )

    # Add true ATE values as text
    for i, model in enumerate(models):
        # Get average y position of bars for this model
        y_pos = np.mean([diffs_all[model][method] for method in methods])
        # Place text above or below depending on if bars are positive or negative
        text_y = -0.005 if y_pos > 0 else 0.005
        ax.text(
            i,
            text_y,
            f"{true_effect[model]:.3f}",
            horizontalalignment="center",
            verticalalignment="top" if y_pos > 0 else "bottom",
            fontsize=12,
        )

    # Add horizontal line at 0 to show no difference from true ATE
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    ax.set_ylabel(f"Difference from True {effect_type}")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    return ax


# %%
models = {
    "base model": {"alpha": [0, 0.5, -0.5, 0], "beta": [-1, 2, 1, -1, 0]},
    "change alpha": {"alpha": [-0.2, 1, -1, 0], "beta": [-1, 2, 1, -1, 0]},
    "reverse treatment effect": {
        "alpha": [0, 0.5, -0.5, 0],
        "beta": [-1, -2, 1, -1, 0],
    },
    "increase treatment effect ": {
        "alpha": [0, 0.5, -0.5, 0],
        "beta": [1, 5, 1, -1, 0],
    },
    "nonlinear treatment": {"alpha": [0, 0.5, -0.5, 1], "beta": [1, -2, 1, -1, 0]},
    "nonlinear outcome": {"alpha": [0, 0.5, -0.5, 0], "beta": [1, -2, 1, -1, 1]},
    "both nonlinear": {"alpha": [0, 0.5, -0.5, 1], "beta": [1, -2, 1, -1, 1]},
}
common_support_threshold = 0.01

# %%
methods = ["TMLE", "AIPW", "IPW"]
true_ate, estimated_ate, std_ate = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATE",
    common_support_threshold=common_support_threshold,
    calibrate=True,
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
plot_causal_effect_estimation_comparison(
    ax,
    true_ate,
    estimated_ate,
    std_ate,
    methods,
    "Estimation Error Compared to True ATE - Logistic Regression Base Estimates",
    "ATE",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## ATT
#

# %%
methods = ["AIPW", "IPW"]
model_kwargs = {"penalty": None, "solver": "lbfgs", "max_iter": 1000}
att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    common_support_threshold=common_support_threshold,
    calibrate=True,
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATT - Logistic Regression Base Estimates",
    "ATT",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## XGboost
#

# %%
from xgboost import XGBClassifier

methods = ["AIPW", "IPW"]
common_support_threshold = 0.001
att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    common_support_threshold=common_support_threshold,
    calibrate=True,
    ps_model=XGBClassifier,
    outcome_model=XGBClassifier,
    ps_model_kwargs={},
    outcome_model_kwargs={},
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATT - XGBoost Base Estimates",
    "ATT",
)
plt.tight_layout()
plt.show()

# %%
methods = [
    "TMLE",
    "AIPW",
    "IPW",
]
common_support_threshold = 0.05
att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATE",
    common_support_threshold=common_support_threshold,
    calibrate=False,
    ps_model=XGBClassifier,
    outcome_model=XGBClassifier,
    ps_model_kwargs={},
    outcome_model_kwargs={},
)

# %%
fig, ax = plt.subplots(figsize=(15, 6))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATE - XGBoost Base Estimates",
    "ATE",
)
plt.tight_layout()
plt.show()
