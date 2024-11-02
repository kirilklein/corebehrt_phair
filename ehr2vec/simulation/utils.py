from typing import Tuple

import numpy as np

from ehr2vec.common.config import Config, get_function


def simulate_outcome(
    proba: np.ndarray, target: np.ndarray, simulation_cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to simulate binary outcome based on provided probabilities and configuration.

    Args:
        proba: Array of predicted probabilities from the model
        target: Array of binary treatment assignments (0/1)
        simulation_cfg: Configuration object containing simulation parameters and function name

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - simulated binary outcomes (0/1)
            - probability of outcome for each instance
    """
    return get_function(simulation_cfg)(proba, target, **simulation_cfg.params)
