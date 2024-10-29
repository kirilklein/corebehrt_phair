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
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
# # !pip install matplotlib
# #!pip install pandas
import pandas as pd
import os
from utils.vis import plot_effect_estimation, plot_propensity_scores
from utils.load import load_estimates


# %% [markdown]
# ## Propensity Scores

# %%
ps = pd.read_csv(
    "/dtu/p1/kirkle/PHAIR_project/corebehrt_phair/pretrain/run_01_continue/finetune_amlodipine_censored_10_hours_pre_amlodipine_followup_start_at_index_date_run_05/predictions_and_targets_calibrated_isotonic.csv"
)

# %%
plot_propensity_scores(ps)

# %%
# fig, ax = plotting.plot_propensity_score_dist(ps, "target", "proba", "propensity_scores.png")
# # ! some issues with the plotting function.

# %% [markdown]
# ## Patient Numbers and Noise Levels

# %%
base_dir = "/dtu/p1/kirkle/PHAIR_project/corebehrt_phair/effect_estimation/synthea/100k/simulated_outcome"
patients_dir = os.path.join(base_dir, "n_patients_amlodipine_a_2p0")
noise_dir = os.path.join(base_dir, "noise_amlodipine_a_2p0")
effect_strength_dir = os.path.join(base_dir, "effect_strength_a")

df_patients = load_estimates(patients_dir, is_noise=False, is_patient_number=True)
df_noise = load_estimates(noise_dir, is_noise=True)
df_effect_strength = load_estimates(
    effect_strength_dir, is_noise=False, is_patient_number=False
)

# %%
df_effect_strength.head()

# %%
df_patients.head()

# %%
# Example usage for patient numbers:
fig, ax = plot_effect_estimation(
    df_patients,
    x_var="patient_number",
    x_label="Number of patients",
    filename="effect_estimation_patients_amlodipine_a_2p0.png",
    xticks=[200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
)
fig


# %%
# Example usage for noise levels:
fig, ax = plot_effect_estimation(
    df_noise,
    x_var="noise_level",
    x_label="Noise level",
    filename="effect_estimation_noise_amlodipine_a_2p0.png",
    xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
)
fig

# %%
# Example usage for patient numbers:
fig, ax = plot_effect_estimation(
    df_effect_strength,
    x_var="effect_counterfactual",
    x_label="True ATE",
    filename="effect_estimation_different_effect_strength_a.png",
    xticks=[-0.25, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
    yticks=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    plot_baseline=False,
)


ax.plot([-0.25, 0.4], [-0.25, 0.4], color="#696969", linestyle="--", label="True ATE")
ax.legend(ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1), loc="upper center")
fig
