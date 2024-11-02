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
