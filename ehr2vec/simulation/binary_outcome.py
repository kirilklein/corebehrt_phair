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
        ps: patient-specific features
        exposure: exposure variable
        a: coefficient for exposure
        b: coefficient for ps
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
    features: np.ndarray,
    exposure: np.ndarray,
    a: float,
    b: float,
    c: float,
    sparsity: float = 0.7,
    scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate outcome from patient embeddings with sparse feature coefficients.

    Args:
        df: DataFrame containing features and exposure
        a: exposure coefficient
        b: coefficient for feature combination
        c: intercept
        sparsity: proportion of features that will have zero coefficients (default: 0.7)
        scale: scale of the normal distribution for feature coefficients (default: 0.1)

    Returns:
        Tuple[np.ndarray, np.ndarray]: binary outcome and probability
    """

    # Generate sparse feature coefficients
    n_features = features.shape[1]
    rng = np.random.RandomState(42)  # Set fixed random state for reproducibility
    W = rng.normal(
        0, scale, size=n_features
    )  # Small coefficients from normal distribution
    zero_mask = rng.random(n_features) < sparsity
    W[zero_mask] = 0  # Set random subset of coefficients to zero

    # Calculate probability using feature combination
    feature_combination = (
        features @ W
    )  # Matrix multiplication of features and coefficients
    probability = sigmoid(a * exposure + b * feature_combination + c)
    binary_outcome = bernoulli.rvs(probability)

    return binary_outcome, probability
