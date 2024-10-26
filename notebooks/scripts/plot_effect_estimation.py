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
    "/dtu/p1/kirkle/PHAIR_project/corebehrt_phair/pretrain/run_01_continue/finetune_EXPOSURE_censored_4_days_pre_EXPOSURE_followup_start_at_index_date_run_04/predictions_and_targets_calibrated_isotonic.csv"
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
patients_dir = os.path.join(base_dir, "n_patients_02")
noise_dir = os.path.join(base_dir, "noise_02")

df_patients = load_estimates(patients_dir, is_noise=False)
df_noise = load_estimates(noise_dir, is_noise=True)

# %%
# Example usage for patient numbers:
plot_effect_estimation(
    df_patients,
    x_var="N",
    x_label="Number of patients",
    filename="effect_estimation_patients_02.png",
    xticks=[200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    yticks=[0, 0.1, 0.2, 0.3, 0.4],
)

# %%
# Example usage for noise levels:
plot_effect_estimation(
    df_noise,
    x_var="noise",
    x_label="Noise level",
    filename="effect_estimation_noise_02.png",
    xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    yticks=[0, 0.1, 0.2, 0.3, 0.4],
)
