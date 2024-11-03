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
from os.path import join
from utils.vis import (
    plot_effect_estimation,
    plot_propensity_scores,
    plot_calibration_curve,
    plot_perfect_calibration,
)
from utils.load import load_estimates
import matplotlib.pyplot as plt


# %% [markdown]
# ## Propensity Scores

# %%
base_dir = "/dtu/p1/kirkle/PHAIR_project/corebehrt_phair/pretrain/run_01_continue"
exp_finetune_dir = "finetune_amlodipine_censored_10_hours_pre_amlodipine_followup_start_at_index_date_run_05"

# %%
ps_calibrated = pd.read_csv(
    join(base_dir, exp_finetune_dir, "predictions_and_targets_calibrated_isotonic.csv")
)
ps_uncalibrated = pd.read_csv(
    join(base_dir, exp_finetune_dir, "predictions_and_targets.csv")
)

fig, ax = plt.subplots()
plot_calibration_curve(ax, ps_calibrated["target"], ps_calibrated["proba"], "Isotonic")
plot_calibration_curve(
    ax, ps_uncalibrated["target"], ps_uncalibrated["proba"], "Uncalibrated"
)
plot_perfect_calibration(ax)
ax.legend()
fig.savefig("propensity_scores_calibration.png")


# %%
plot_propensity_scores(ps_calibrated)
# plot_propensity_scores(ps_uncalibrated)

# %% [markdown]
# ## Outcome Probas

# %%
outcome_probas_dir = "finetune_SIMULATED_censored_1_hours_post_amlodipine_followup_start_at_index_date_a_6p0_b_2p0_c_neg2p0"
outcome_probas = pd.read_csv(
    join(base_dir, outcome_probas_dir, "predictions_and_targets.csv")
)
outcome_probas_calibrated = pd.read_csv(
    join(
        base_dir, outcome_probas_dir, "predictions_and_targets_calibrated_isotonic.csv"
    )
)

fig, ax = plt.subplots()
plot_calibration_curve(
    ax, outcome_probas["target"], outcome_probas["proba"], "Uncalibrated"
)
plot_calibration_curve(
    ax,
    outcome_probas_calibrated["target"],
    outcome_probas_calibrated["proba"],
    "Isotonic",
)
plot_perfect_calibration(ax)
ax.legend()
fig.savefig("outcome_probas_calibration.png")
plot_propensity_scores(outcome_probas_calibrated)

# %% [markdown]
# ## Counterfactuals

# %%
counterfactuals = pd.read_csv(
    join(
        base_dir,
        outcome_probas_dir,
        "counterfactual_predictions_a_6p0_b_2p0_c_neg2p0/counterfactual_predictions.csv",
    )
)
counterfactuals = counterfactuals.merge(outcome_probas[["pid", "target"]], on="pid")
counterfactuals.target = counterfactuals.target.apply(lambda x: 0 if x == 1 else 1)
counterfactuals.head()
plot_propensity_scores(counterfactuals)
# countefactual predictions are totally off. maybe we should sample from the distribution of the codes + other codes influence the outcome
# often changing one code doesn't even make sense. Should we train the model on counterfactuals?

# %% [markdown]
# ## Patient Numbers and Noise Levels

# %%
effect_estimation_base_dir = "/dtu/p1/kirkle/PHAIR_project/corebehrt_phair/effect_estimation/synthea/100k/simulated_outcome"

patients_dir = os.path.join(
    effect_estimation_base_dir, "n_patients", "n_patients_amlodipine_a_neg1p0"
)
noise_dir = os.path.join(
    effect_estimation_base_dir, "noise", "noise_amlodipine_a_neg1p0"
)
effect_strength_dir = os.path.join(
    effect_estimation_base_dir, "effect_strength", "effect_strength_a_x_b_2p0_c_neg2p0"
)

df_patients = load_estimates(patients_dir, is_noise=False, is_patient_number=True)
df_noise = load_estimates(noise_dir, is_noise=True)
df_effect_strength = load_estimates(
    effect_strength_dir, is_noise=False, is_patient_number=False
)

# %%
# Example usage for patient numbers:
fig, ax = plot_effect_estimation(
    df_patients,
    x_var="patient_number",
    x_label="Number of patients",
    filename="effect_estimation_patients_amlodipine_a_2p0.png",
    xticks=[200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 14000],
    yticks=[
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        0.1,
    ],
)
fig


# %%
# Example usage for noise levels:
fig, ax = plot_effect_estimation(
    df_noise,
    x_var="noise_level",
    x_label="Noise level",
    filename="effect_estimation_noise_amlodipine_a_2p0.png",
    xticks=[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
    yticks=[-0.3, -0.2, -0.1, 0, 0.1],
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
fig.savefig("figures/effect_estimation_different_effect_strength_a.png")
fig


# %% [markdown]
# ## Use the simulation model as outcome model

# %%
effect_strength_dir = os.path.join(
    effect_estimation_base_dir,
    "effect_strength",
    "effect_strength_a_x_b_2p0_c_neg2p0_exact_outcome_model",
)
df_effect_strength_exact = load_estimates(
    effect_strength_dir, is_noise=False, is_patient_number=False
)
fig, ax = plot_effect_estimation(
    df_effect_strength_exact,
    x_var="effect_counterfactual",
    x_label="True ATE",
    filename="effect_estimation_different_effect_strength_a_exact_outcome_model.png",
    xticks=[
        -0.05,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ],
    yticks=[
        -0.05,
        0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ],
    plot_baseline=False,
)
ax.plot([-0.05, 0.95], [-0.05, 0.95], color="#696969", linestyle="--", label="True ATE")
fig.savefig(
    "figures/effect_estimation_different_effect_strength_a_exact_outcome_model.png"
)
fig
