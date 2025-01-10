import numpy as np
import pandas as pd
from ehr2vec.common.default_args import ORG_PID_COL, PROBA_COL, TARGET_COL


def save_probas_and_targets(
    pids: np.ndarray,
    binary_outcome: np.ndarray,
    probability: np.ndarray,
    output_path: str,
) -> None:
    """
    Save predicted probabilities and binary outcomes to a CSV file.

    Args:
        pids: Array of patient IDs
        binary_outcome: Array of binary outcomes (0/1)
        probability: Array of predicted probabilities
        output_path: Path where the CSV file will be saved
    """
    pd.DataFrame(
        {ORG_PID_COL: pids, TARGET_COL: binary_outcome, PROBA_COL: probability}
    ).to_csv(output_path, index=False)


def save_counterfactual_probas_and_targets(
    pids: np.ndarray,
    target: np.ndarray,
    binary_outcome_exp: np.ndarray,
    binary_outcome_ctrl: np.ndarray,
    probability_exp: np.ndarray,
    probability_ctrl: np.ndarray,
    output_path: str,
) -> None:
    """
    Save counterfactual predictions and outcomes to a CSV file.

    Args:
        pids: Array of patient IDs
        target: Array of original treatment assignments
        binary_outcome_exp: Array of binary outcomes under exposure/treatment
        binary_outcome_ctrl: Array of binary outcomes under control
        probability_exp: Array of predicted probabilities under exposure/treatment
        probability_ctrl: Array of predicted probabilities under control
        output_path: Path where the CSV file will be saved
    """
    pd.DataFrame(
        {
            ORG_PID_COL: pids,
            TARGET_COL: select_counterfactual(
                target, binary_outcome_exp, binary_outcome_ctrl
            ),
            PROBA_COL: select_counterfactual(target, probability_exp, probability_ctrl),
        }
    ).to_csv(output_path, index=False)


def select_counterfactual(
    target: np.ndarray, exp_value: np.ndarray, ctrl_value: np.ndarray
) -> np.ndarray:
    """
    Select counterfactual values based on original treatment assignment.

    Args:
        target: Array of original treatment assignments (0/1)
        exp_value: Array of values under exposure/treatment
        ctrl_value: Array of values under control

    Returns:
        np.ndarray: Array of counterfactual values - returns exposure values for control group
                   and control values for exposure group
    """
    return np.where(target == 0, exp_value, ctrl_value)
