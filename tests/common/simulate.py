import numpy as np
import pandas as pd

from ehr2vec.common.default_args import (
    CF_CONTROL_COL,
    CF_TREATED_COL,
    OUTCOME_COL,
    OUTCOME_PROBABILITY_COL,
    PID_COL,
    PS_COL,
    TREATMENT_COL,
)


def simulate_binary_data_complex(
    n: int, alpha: list, beta: list, seed=None
) -> pd.DataFrame:
    """
    Simulate binary outcome data with covariates and binary treatment.

    Args:
        n: Number of samples
        alpha: Treatment model coefficients
        beta: Outcome model coefficients
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    # Simulate covariates - only 3 features needed based on usage
    X = rng.normal(0, 1, (n, 3))

    # Pad coefficients with zeros if needed
    alpha = np.pad(alpha, (0, 6 - len(alpha)), mode="constant")
    beta = np.pad(beta, (0, 8 - len(beta)), mode="constant")

    # Treatment model
    # Treatment model with nonlinear terms
    logit_p = (
        alpha[0]
        + np.sum(
            [alpha[i + 1] * X[:, i] for i in range(3)], axis=0
        )  # Only use 3 features
        + alpha[1] * X[:, 0] ** 2
        + alpha[2] * np.sin(X[:, 1])
        + alpha[3] * np.exp(X[:, 2] / 2)
        + alpha[4] * np.tanh(X[:, 1])
        + alpha[5] * X[:, 0] * X[:, 1]
    )  # Interaction term
    p = 1 / (1 + np.exp(-logit_p))
    A = rng.binomial(1, p)

    # Outcome model with nonlinear terms
    logit_q = (
        beta[0]
        + beta[1] * A
        + beta[2] * X[:, 0] ** 3
        + beta[3] * np.cos(X[:, 1])
        + beta[4] * A * X[:, 0] * X[:, 1]
        + beta[5] * np.exp(X[:, 2] / 3)
        + beta[6] * np.log(np.abs(X[:, 1]) + 1)
    )
    q = 1 / (1 + np.exp(-logit_q))
    Y = rng.binomial(1, q)

    # Counterfactual outcome
    logit_q_cf = (
        beta[0]
        + beta[1] * (1 - A)
        + beta[2] * X[:, 0] ** 3
        + beta[3] * np.cos(X[:, 1])
        + beta[4] * (1 - A) * X[:, 0] * X[:, 1]  # Fixed coefficient index
        + beta[5] * np.exp(X[:, 2] / 3)
        + beta[6] * np.log(np.abs(X[:, 1]) + 1)  # Fixed coefficient index
    )
    q_cf = 1 / (1 + np.exp(-logit_q_cf))
    Y_cf = rng.binomial(1, q_cf)

    # Create dataframe - only create 3 X columns since that's what we generate
    data = pd.DataFrame(
        {f"X{i+1}": X[:, i] for i in range(3)} | {"A": A, "Y": Y, "Y_cf": Y_cf}
    )

    return data


def create_synthetic_test_data(n: int = 5) -> pd.DataFrame:
    """Create a tiny DataFrame with T, Y, PS, and some CF columns for testing."""
    # For reproducibility
    np.random.seed(42)
    df = pd.DataFrame(
        {
            PID_COL: range(n),
            TREATMENT_COL: np.random.binomial(1, 0.5, n),
        }
    )

    # Generate PS with slight shift for treated individuals
    base_ps = np.random.beta(2, 3, n)  # base propensity scores
    ps_shift = 0.1  # shift amount for treated individuals
    df[PS_COL] = np.where(
        df[TREATMENT_COL] == 1,
        np.minimum(base_ps + ps_shift, 1),  # ensure PS doesn't exceed 1
        base_ps,
    )

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Create base probability and treatment effect in logit space
    base_prob = np.random.beta(2, 2, n)  # baseline probability
    treatment_logit = np.log(1.4 / 0.6)  # logit equivalent of 0.4 treatment effect

    # Calculate outcome probability using logit space
    df[OUTCOME_PROBABILITY_COL] = sigmoid(
        np.log(base_prob / (1 - base_prob)) + df[TREATMENT_COL] * treatment_logit
    )

    # Simulate outcome based on treatment-adjusted probability
    df[OUTCOME_COL] = np.random.binomial(1, df[OUTCOME_PROBABILITY_COL], n)

    # Simulate counterfactual outcomes under treatment and control
    df[CF_TREATED_COL] = sigmoid(np.log(base_prob / (1 - base_prob)) + treatment_logit)
    df[CF_CONTROL_COL] = sigmoid(np.log(base_prob / (1 - base_prob)))

    return df
