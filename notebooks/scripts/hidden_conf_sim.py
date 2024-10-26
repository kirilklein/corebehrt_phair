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

# %% [markdown]
# Here is the updated markdown description in the original format with LaTeX code enclosed by `$$`:
#
# ---
#
# ## Simulation of Hidden Confounders and Conditional Effect
#
# This notebook demonstrates the simulation of data under the presence of a hidden confounder and calculates the effect of \(X\) on \(Y\) both from the structural causal model and from observational data.
#
# ### Step 1: Define a Simple Model
#
# We assume the following structural causal model (SCM) with a hidden confounder \(U_{XY}\):
#
# - $U_{XY} \sim \mathcal{N}(0, 1)$
# - $C = \alpha_C U_{XY} + \epsilon_C$, where $\epsilon_C \sim \mathcal{N}(0, \sigma_C)$
# - $X = \sigma(\alpha_X U_{XY} + \gamma_{CX} C)$, where $\sigma$ is the logistic function
# - $Y = \sigma(\beta X + \alpha_Y U_{XY} + \gamma_{CY} C)$, where $\sigma$ is the logistic function
#
# Here:
#
# - $\alpha_C, \alpha_X, \alpha_Y, \gamma_{CX}, \gamma_{CY}, \beta$ are coefficients that define the strength of the relationships, where $\beta$ is the log odds ratio of $Y$ given $X$.
# - $\sigma_C$ is the standard deviation of the noise in the equation for $C$.
#

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


class Model:
    def __init__(
        self,
        alpha_C=1,
        alpha_X=1,
        alpha_Y=1,
        beta=2.0,
        gamma_CX=0,
        sigma_C=0.5,
        gamma_CY=0,
    ):
        """
        Initializes the structural causal model.
        Parameters:
        alpha_C, alpha_X, alpha_Y (float): Coefficients for the unobserved confounder in the equations for C and Y.
        gamma_CX, gamma_CY (float): Coefficients for C in the equations for X and Y.
        beta (float): Coefficient for X in the equation for Y. (log odds ratio of Y given X)
        sigma_C (float): Standard deviations of the noise in the equations for C, X, and Y.
        """
        self.alpha_C = alpha_C
        self.alpha_X = alpha_X
        self.alpha_Y = alpha_Y
        self.gamma_CX = gamma_CX
        self.gamma_CY = gamma_CY
        self.beta = beta
        self.sigma_C = sigma_C

    def simulate_binary_data(self, n=1000):
        """
        Simulates data with binary X and Y based on the specified structural causal model.

        Parameters:
        n (int): Number of samples to generate.

        Returns:
        pd.DataFrame: Simulated data with binary 'X' and 'Y'
        """
        # Generate unobserved confounder
        U_XY = np.random.normal(0, 1, n)

        # Generate observed covariate C
        C = self.alpha_C * U_XY + np.random.normal(0, self.sigma_C, n)

        # Generate binary X using logistic function
        logit_X = self.alpha_X * U_XY + self.gamma_CX * C
        p_X = expit(logit_X)  # expit is the logistic sigmoid function
        X = np.random.binomial(1, p_X)

        # Generate binary Y using logistic function
        logit_Y = self.beta * X + self.alpha_Y * U_XY + self.gamma_CY * C
        p_Y = expit(logit_Y)
        Y = np.random.binomial(1, p_Y)

        return pd.DataFrame({"X": X, "Y": Y, "C": C})


# %%
from sklearn.neighbors import NearestNeighbors


def matching(data, adjustment_variable=None):
    adjustment_data = data[[adjustment_variable]]
    logit = LogisticRegression()
    logit.fit(adjustment_data, data["X"])
    data["propensity_score"] = logit.predict_proba(adjustment_data)[:, 1]

    treated = data[data["X"] == 1]
    control = data[data["X"] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity_score"]])
    distances, indices = nn.kneighbors(treated[["propensity_score"]])
    matched_control_indices = indices.flatten()
    matched_control = control.iloc[matched_control_indices].reset_index()
    return treated, matched_control


def calculate_log_odds_ratio(data, adjustment_variable=None):
    """
    Estimates the effect of a binary X on a binary Y using non-parametric methods.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'X' and 'Y'.

    Returns:
    dict: Estimated difference in proportions and odds ratio
    """
    if adjustment_variable is not None:
        treated, controls = matching(data, adjustment_variable)
    else:
        treated = data[data["X"] == 1]
        controls = data[data["X"] == 0]
    # Calculate proportions
    treated_outcome_rate = treated["Y"].mean()
    control_outcome_rate = controls["Y"].mean()

    # Odds ratio
    treated_odds = treated_outcome_rate / (1 - treated_outcome_rate)
    control_odds = control_outcome_rate / (1 - control_outcome_rate)
    odds_ratio = treated_odds / control_odds

    return np.log(odds_ratio)


# %%
LOR = 2
model = Model(
    alpha_C=0, alpha_X=0, alpha_Y=0, gamma_CX=0, gamma_CY=0, beta=LOR, sigma_C=0
)
data = model.simulate_binary_data(100000)
estimated_lor = calculate_log_odds_ratio(data)

print(f"Estimated LOR {estimated_lor} (True LOR: 2.0) in the absence of confounding")

# %% [markdown]
# ## Measure the Effect of $X$ on $Y$
# ### Strong effect of $U_{XY}$ on $C$

# %%
ALPHA_C = 10  # How strongly is C influenced by the unobserved confounder
models = {
    "C": Model(alpha_C=ALPHA_C),
    "CX": Model(alpha_C=ALPHA_C, gamma_CX=1),
    "CY": Model(alpha_C=ALPHA_C, gamma_CY=1),
    "CXY": Model(alpha_C=ALPHA_C, gamma_CX=1, gamma_CY=1),
}

data = {name: model.simulate_binary_data(100000) for name, model in models.items()}
lor_w_adj = {
    name: calculate_log_odds_ratio(data, adjustment_variable="C")
    for name, data in data.items()
}
lor_wo_adj = {name: calculate_log_odds_ratio(data) for name, data in data.items()}

print(f"True LOR: {models['C'].beta}")
print(f"LOR with adjustment: {lor_w_adj}")
print(f"LOR withot adjustment: {lor_wo_adj}")

# %% [markdown]
# ### Weak effect of $U_{XY}$ on $C$

# %%
ALPHA_C = 1  # How strongly is C influenced by the unobserved confounder
models = {
    "C": Model(alpha_C=ALPHA_C),
    "CX": Model(alpha_C=ALPHA_C, gamma_CX=1),
    "CY": Model(alpha_C=ALPHA_C, gamma_CY=1),
    "CXY": Model(alpha_C=ALPHA_C, gamma_CX=1, gamma_CY=1),
}

data = {name: model.simulate_binary_data(100000) for name, model in models.items()}
lor_w_adj = {
    name: calculate_log_odds_ratio(data, adjustment_variable="C")
    for name, data in data.items()
}
lor_wo_adj = {name: calculate_log_odds_ratio(data) for name, data in data.items()}

print(f"True LOR: {models['C'].beta}")
print(f"LOR with adjustment: {lor_w_adj}")
print(f"LOR withot adjustment: {lor_wo_adj}")
