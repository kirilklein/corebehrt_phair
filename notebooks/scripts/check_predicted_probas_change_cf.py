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
def simulate_binary_data_complex(
    n: int, alpha: list, beta: list, seed=None
) -> pd.DataFrame:
    """
    Simulate simple binary outcome data with two covariates and
    a binary treatment simulated from a logistic regression model.
    n: number of samples
    """
    if seed is not None:
        # ise new generator with seed
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    # Simulate covariates
    X = rng.normal(0, 1, (n, 6))
    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]
    X4 = X[:, 3]
    X5 = X[:, 4]
    X6 = X[:, 5]

    # alpha can be a list of length 3. Extend to length 6 with 0s
    if len(alpha) < 6:
        alpha = np.pad(alpha, (0, 6 - len(alpha)), mode="constant")
    # Simulate treatment
    logit_p = (
        alpha[0]
        + alpha[1] * X1
        + alpha[2] * X2
        + alpha[3] * X1 * X2
        + alpha[4] * X1**2 * X2**2
        + alpha[5] * X2**2 * X3**2
        + alpha[4] * X3**2 * X4**2
        + alpha[5] * X4**2 * X5**2
        + alpha[3] * X3 * X4 * X5 * X6
        + alpha[4] * X5**4 * X6**3
        + alpha[5] * X6**3
        + alpha[3] * X5 * X6 * X1**2 * X2**2
    )

    p = 1 / (1 + np.exp(-logit_p))  # Using numpy's logistic function directly
    A = rng.binomial(1, p)

    if len(beta) < 8:
        beta = np.pad(beta, (0, 8 - len(beta)), mode="constant")
    # Simulate outcome
    logit_q = (
        beta[0]
        + beta[1] * A
        + beta[2] * X1
        + beta[3] * X2
        + beta[4] * X1 * X2
        + beta[5] * X1**2
        + beta[6] * X2**2
        + beta[7] * A**2
    )
    q = 1 / (1 + np.exp(-logit_q))
    Y = rng.binomial(1, q)

    logit_q_cf = (
        beta[0] + beta[1] * (1 - A) + beta[2] * X1 + beta[3] * X2 + beta[4] * X1 * X2
    )
    q_cf = 1 / (1 + np.exp(-logit_q_cf))
    Y_cf = rng.binomial(1, q_cf)

    data = pd.DataFrame(
        {
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "X4": X4,
            "X5": X5,
            "X6": X6,
            "A": A,
            "Y": Y,
            "Y_cf": Y_cf,
        }
    )

    return data


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

plt.xlabel("Prediction Difference (Original - Counterfactual)")
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

plt.xlabel("Prediction Difference (Original - Counterfactual)")
plt.ylabel("Density")
plt.title("Distribution of Prediction Changes from Counterfactual")
plt.legend(title="True Treatment Status")
plt.show()
