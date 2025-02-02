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
from CausalEstimate.simulation.binary_simulation import (
    compute_ATE_theoretical_from_data,
    compute_ATT_theoretical_from_data,
    simulate_binary_data,
)
from sklearn.ensemble import RandomForestClassifier
from tests.common.estimates import (
    compute_ATE_from_data,
    compute_ATT_from_data,
    estimate_causal_effects_with_multiple_methods,
)
from tests.common.plot_sim import (
    plot_calibration_curves,
    plot_causal_effect_estimation_comparison,
    plot_propensity_score_dist_for_models,
)

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
    "azure exp. mock": {
        "alpha": [0, -0.1, 0.1, -0.5, 0.5, -0.5],
        "beta": [0, 4, 0, 0],
    },  # extremely nonlinear treatment assignment but outcome only linearly dependent on A
}
common_support_threshold = 0.01

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
    n_samples=10000,
    effect_type="ATE",
    common_support_threshold=common_support_threshold,
)

# %%
true_ate

# %%
estimated_ate

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
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()
fig.savefig("figures/ATE_comparison_LR.png", dpi=300)


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
    n_samples=10000,
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
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()
fig.savefig("figures/ATE_comparison_RF.png", dpi=300)

# %% [markdown]
# ### ATT
#

# %%
methods = ["AIPW", "IPW"]

att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    n_samples=10000,
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
    "Estimation Error Compared to True ATT - RF",
    "ATT",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()

fig.savefig("figures/ATT_comparison_RF.png", dpi=300)

# %%
methods = ["AIPW", "IPW"]

att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    n_samples=10000,
    common_support_threshold=common_support_threshold,
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
    "Estimation Error Compared to True ATT - ps model LR/ outcome model RF",
    "ATT",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()

fig.savefig("figures/ATT_comparison_ps_LR_out_RF.png", dpi=300)

# %%
methods = ["AIPW", "IPW"]

att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    n_samples=10000,
    common_support_threshold=common_support_threshold,
    ps_model_dict=rf_dic,
)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATT - ps model RF/ outcome model LR",
    "ATT",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()

fig.savefig("figures/ATT_comparison_ps_RF_out_LR.png", dpi=300)

# %%
methods = ["AIPW", "IPW"]

att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATT",
    n_samples=10000,
    common_support_threshold=common_support_threshold,
)

# %%
fig, ax = plt.subplots(figsize=(12, 5))
plot_causal_effect_estimation_comparison(
    ax,
    att_true,
    att_estimates,
    att_stds,
    methods,
    "Estimation Error Compared to True ATT - LR",
    "ATT",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()

fig.savefig("figures/ATT_comparison_LR.png", dpi=300)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Given data
true_effect = 0.484
methods = ["True ATE", "AIPW", "IPW", "TMLE"]
effects = [0.484, 0.245, 0.481, 0.302]
stds = [0.0, 0.004, 0.008, 0.007]

# Compute differences from the true effect
diffs = [e - true_effect for e in effects]
diffs = diffs[1:]
stds = stds[1:]
methods = methods[1:]
# Colors for each method
colors = [
    "#2ecc71",  # Emerald green (True ATE)
    "#3498db",  # Dodger blue (AIPW)
    "#e74c3c",  # Alizarin red (IPW)
    "#9b59b6",  # Amethyst purple (TMLE)
]

fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(methods))

ax.bar(x, diffs, yerr=stds, align="center", alpha=0.8, capsize=5, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.set_ylabel("Difference from True Effect")
ax.set_title("Difference Between Estimated and True ATE")
ax.axhline(0, color="black", linewidth=1)

# Optionally add numeric labels above bars
for i, (diff, std) in enumerate(zip(diffs, stds)):
    ax.text(
        i,
        diff + (0.01 if diff >= 0 else -0.01),
        f"{diff:.3f}",
        ha="center",
        va="bottom" if diff >= 0 else "top",
        fontsize=9,
    )

plt.tight_layout()
plt.show()
fig.savefig("figures/ATE_comparison_azure.png", dpi=300)


# %%
# Given data
true_effect = 0.486
methods = ["True ATT", "AIPW", "IPW"]
effects = [0.486, 0.184, 0.478]
stds = [0.0, 0.004, 0.004]

# Compute differences from the true effect
diffs = [e - true_effect for e in effects]
diffs = diffs[1:]
stds = stds[1:]
methods = methods[1:]
# Colors for each method
colors = [
    "#3498db",  # Dodger blue (AIPW)
    "#e74c3c",  # Alizarin red (IPW)
]

fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(methods))

ax.bar(x, diffs, yerr=stds, align="center", alpha=0.8, capsize=5, color=colors)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.set_ylabel("Difference from True Effect")
ax.set_title("Difference Between Estimated and True ATT")
ax.axhline(0, color="black", linewidth=1)

# Optionally add numeric labels above bars
for i, (diff, std) in enumerate(zip(diffs, stds)):
    ax.text(
        i,
        diff + (0.01 if diff >= 0 else -0.01),
        f"{diff:.3f}",
        ha="center",
        va="bottom" if diff >= 0 else "top",
        fontsize=9,
    )

plt.tight_layout()
plt.show()
fig.savefig("figures/ATT_comparison_azure.png", dpi=300)


# %%
methods = [
    "TMLE",
    "AIPW",
    "IPW",
]

att_true, att_estimates, att_stds = estimate_causal_effects_with_multiple_methods(
    models,
    methods,
    effect_type="ATE",
    n_samples=10000,
    common_support_threshold=common_support_threshold,
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
    "Estimation Error Compared to True ATE - ps model RF/ outcome model LR",
    "ATE",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()

fig.savefig("figures/ATE_comparison_ps_RF_out_LR.png", dpi=300)

# %% [markdown]
# # New Experiments on Azure
#

# %%
MODEL1 = "model_3_0_-3"
MODEL2 = "model_1_0.2_-1"
MODEL3 = "model_1_0.5_-1"
MODEL4 = "model_1_1_-2"
MODEL5 = "model_1_2_-3"
true_ate = {
    MODEL1: 0.4560,
    MODEL2: 0.2322,
    MODEL3: 0.2318,
    MODEL4: 0.0782,
    MODEL5: 0.0782,
}

estimated_ate = {
    MODEL1: {"TMLE": 0.4595, "AIPW": 0.4592, "IPW": 0.4603},
    MODEL2: {"TMLE": 0.2329, "AIPW": 0.2286, "IPW": 0.2318},
    MODEL3: {"TMLE": 0.2346, "AIPW": 0.2293, "IPW": 0.2325},
    MODEL4: {"TMLE": 0.0713, "AIPW": 0.0696, "IPW": 0.0704},
    MODEL5: {"TMLE": 0.0713, "AIPW": 0.0696, "IPW": 0.0704},
}

estimated_ate_std = {
    MODEL1: {"TMLE": 0.0044, "AIPW": 0.0044, "IPW": 0.0055},
    MODEL2: {"TMLE": 0.0073, "AIPW": 0.0100, "IPW": 0.0098},
    MODEL3: {"TMLE": 0.0048, "AIPW": 0.0082, "IPW": 0.0071},
    MODEL4: {"TMLE": 0.0038, "AIPW": 0.0040, "IPW": 0.0041},
    MODEL5: {"TMLE": 0.0038, "AIPW": 0.0040, "IPW": 0.0041},
}


# %%
fig, ax = plt.subplots(figsize=(12, 5))
methods = ["TMLE", "AIPW", "IPW"]
plot_causal_effect_estimation_comparison(
    ax,
    true_ate,
    estimated_ate,
    estimated_ate_std,
    methods,
    "Estimation Error Compared to True ATE: Real EHR",
    "ATE",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()
fig.savefig("figures/ATE_comparison_z_ts_sim.png", dpi=300)

# %% [markdown]
# ### Extended model with outcome contributing to simulation

# %%
MODEL1 = "a1_b0.5_c1_d-3"
MODEL2 = "a1_b1_c0.5_d-3"
MODEL3 = "a1_b1_c2_d-3"
MODEL4 = "a1_b1_c1_d-3"
true_ate = {MODEL1: 0.0989, MODEL2: 0.0876, MODEL3: 0.1152, MODEL4: 0.1009}

estimated_ate = {
    MODEL1: {"TMLE": 0.0957, "AIPW": 0.0934, "IPW": 0.0940},
    MODEL2: {"TMLE": 0.0889, "AIPW": 0.0884, "IPW": 0.0884},
    MODEL3: {"TMLE": 0.1170, "AIPW": 0.1124, "IPW": 0.1135},
    MODEL4: {"TMLE": 0.1091, "AIPW": 0.1071, "IPW": 0.1085},
}

estimated_ate_std = {
    MODEL1: {"TMLE": 0.0042, "AIPW": 0.0047, "IPW": 0.0044},
    MODEL2: {"TMLE": 0.0040, "AIPW": 0.0047, "IPW": 0.0043},
    MODEL3: {"TMLE": 0.0061, "AIPW": 0.0080, "IPW": 0.0074},
    MODEL4: {"TMLE": 0.0039, "AIPW": 0.0040, "IPW": 0.0041},
}


# %%
fig, ax = plt.subplots(figsize=(12, 5))
methods = ["TMLE", "AIPW", "IPW"]
plot_causal_effect_estimation_comparison(
    ax,
    true_ate,
    estimated_ate,
    estimated_ate_std,
    methods,
    "Estimation Error Compared to True ATE: Real EHR",
    "ATE",
)
ax.set_ylim(-0.1, 0.1)
plt.tight_layout()
plt.show()
fig.savefig("figures/ATE_comparison_zt_zo_sim.png", dpi=300)
