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
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import logistic

if ".." not in sys.path:
    sys.path.append("..")
from CausalEstimate.simulation.binary_simulation import simulate_binary_data
from tests.common.predictions import predict_and_save

# %%

# %%
MODELS = {
    "azure exp. mock": {
        "alpha": [0, -0.2, 0.2, -0.6, 0.7, -0.8],
        "beta": [0, 4, 0, 0],
    },  # extremely nonlinear treatment assignment but outcome only linearly dependent on A
}

# %% [markdown]
# ## Standard sandbox model

# %%
save_dir = "tmp"
model = MODELS["azure exp. mock"]
data = simulate_binary_data(10000, model["alpha"], model["beta"], seed=41)
data["pid"] = range(len(data))
predict_and_save(data, save_dir, seed=41)

# %%
cf_pred = pd.read_csv(join(save_dir, "test_data", "counterfactual_predictions.csv"))
pred = pd.read_csv(join(save_dir, "test_data", "outcome_predictions.csv"))
ps = pd.read_csv(join(save_dir, "test_data", "ps_scores.csv"))
df = pd.merge(cf_pred, pred, on="pid", suffixes=("_cf", "_pred"))
df = pd.merge(df, ps, on="pid").rename(
    columns={"proba": "proba_ps", "target": "target_ps"}
)
df.head()

# %%
df["pred_diff"] = df["proba_cf"] - df["proba_pred"]

# Create histogram colored by target_ps
plt.figure(figsize=(10, 6))
for target in [0, 1]:
    mask = df["target_ps"] == target
    plt.hist(
        df[mask]["pred_diff"],
        bins=50,
        alpha=0.5,
        color="magenta" if target == 1 else "skyblue",
        label="treated" if target == 1 else "control",
    )

plt.xlabel("Prediction Difference (Counterfactual - Original)")
plt.ylabel("Density")
plt.title("Distribution of Prediction Changes from Counterfactual")
plt.legend(title="True Treatment Status")
plt.show()


# %% [markdown]
# ## More complex treatment model

# %%
alpha = [0, 0.5, -0.5, 1, -1, 0.5, 0.25, -0.25, 0.1, -0.1, 0.5]
data = simulate_binary_data_complex(10000, alpha, model["beta"], seed=41)
data["pid"] = range(len(data))
predict_and_save(data, save_dir, seed=41)
cf_pred = pd.read_csv(join(save_dir, "test_data", "counterfactual_predictions.csv"))
pred = pd.read_csv(join(save_dir, "test_data", "outcome_predictions.csv"))
ps = pd.read_csv(join(save_dir, "test_data", "ps_scores.csv"))
df = pd.merge(cf_pred, pred, on="pid", suffixes=("_cf", "_pred"))
df = pd.merge(df, ps, on="pid").rename(
    columns={"proba": "proba_ps", "target": "target_ps"}
)
df.head()
df["pred_diff"] = df["proba_cf"] - df["proba_pred"]


# %%
df["pred_diff"] = df["proba_cf"] - df["proba_pred"]

# Create histogram colored by target_ps
plt.figure(figsize=(10, 6))
for target in [0, 1]:
    mask = df["target_ps"] == target
    plt.hist(
        df[mask]["pred_diff"],
        bins=50,
        alpha=0.5,
        color="magenta" if target == 1 else "skyblue",
        label="treated" if target == 1 else "control",
    )

plt.xlabel("Prediction Difference (Counterfactual - Original)")
plt.ylabel("Density")
plt.title("Distribution of Prediction Changes from Counterfactual")
plt.legend(title="True Treatment Status")
plt.show()
