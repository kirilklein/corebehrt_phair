from typing import Tuple

import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import bernoulli


def tbehrt(
    propensity_score: np.ndarray,
    exposure: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate binary outcome as presented in
    Rao, Shishir, et al.
    "Targeted-BEHRT: deep learning for observational causal inference on longitudinal electronic health records."
    IEEE Transactions on Neural Networks and Learning Systems 35.4 (2022): 5027-5038.

    Args:
        propensity_score: patient-specific features
        exposure: exposure variable
        a: coefficient for exposure
        b: coefficient for propenrity score
        c: coefficient for interaction term

    Returns:
        Tuple[np.ndarray, np.ndarray]: binary outcome and probability
    """
    probability = sigmoid(a * exposure + b * (propensity_score + c))
    binary_outcome = bernoulli.rvs(probability)
    return binary_outcome, probability


def linear_logistic(
    propensity_score: np.ndarray,
    exposure: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    probability = sigmoid(a * exposure + b * propensity_score + c)
    binary_outcome = bernoulli.rvs(probability)
    return binary_outcome, probability


def simulate_outcome_from_embeddings(
    z_t: np.ndarray,
    z_o: np.ndarray,
    exposure: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
    t_sparsity: float = 0.7,
    o_sparsity: float = 0.7,
    t_scale: float = 0.1,
    o_scale: float = 0.1,
    return_coefficients: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate outcome from treatment and outcome patient embeddings with sparse feature coefficients.

    Args:
        z_t: treatment patient embeddings
        z_o: outcome patient embeddings
        exposure: exposure variable
        a: exposure coefficient
        b: treatment patient embeddings coefficient
        c: outcome patient embeddings coefficient
        d: intercept
        t_sparsity: proportion of treatment patient embeddings that will have zero coefficients (default: 0.7)
        o_sparsity: proportion of outcome patient embeddings that will have zero coefficients (default: 0.7)
        t_scale: scale of the normal distribution for treatment patient embeddings coefficients (default: 0.1)
        o_scale: scale of the normal distribution for outcome patient embeddings coefficients (default: 0.1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: binary outcome and probability
    """

    # Generate sparse feature coefficients
    n_t = z_t.shape[1]
    n_o = z_o.shape[1]
    rng = np.random.RandomState(42)  # Set fixed random state for reproducibility
    W_t = rng.normal(
        0, t_scale, size=n_t
    )  # Small coefficients from normal distribution
    W_o = rng.normal(
        0, o_scale, size=n_o
    )  # Small coefficients from normal distribution
    zero_mask_t = rng.random(n_t) < t_sparsity
    zero_mask_o = rng.random(n_o) < o_sparsity

    W_t[zero_mask_t] = 0  # Set random subset of coefficients to zero
    W_o[zero_mask_o] = 0  # Set random subset of coefficients to zero

    # Calculate probability using feature combination
    s_t = z_t @ W_t
    s_o = z_o @ W_o
    probability = sigmoid(a * exposure + b * s_t + c * s_o + d)
    binary_outcome = bernoulli.rvs(probability)
    if return_coefficients:
        return binary_outcome, probability, W_t, W_o
    return binary_outcome, probability
