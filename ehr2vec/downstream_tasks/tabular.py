from os.path import join

import numpy as np
import torch
import pandas as pd


class TabularData:
    """
    Container for storing feature vectors, exposures, targets, and patient IDs
    for a single fold and mode (e.g., train or validation).
    """

    def __init__(
        self,
        vectors: torch.Tensor,
        exposures: np.ndarray,
        targets: np.ndarray,
        pids: torch.Tensor,
    ):
        self.vectors = vectors
        self.exposures = exposures
        self.targets = targets
        self.pids = pids


def load_tabular_data(
    fold_dir: str, mode: str, outcomes: pd.Series, exposures: pd.Series
) -> TabularData:
    """
    Load vectors, exposures, and targets for a specific fold and mode.

    Args:
        fold_dir (str): Directory path for the fold.
        mode (str): 'train' or 'val'.
        outcomes (pd.Series): Outcome labels indexed by patient ID.
        exposures (pd.Series): Exposure labels indexed by patient ID.

    Returns:
        TabularData: Container with vectors, exposures, targets, and pids.
    """
    # Load vectors
    vectors = torch.load(join(fold_dir, f"{mode}_patient_vectors.pt"))
    # Load PIDs
    pids = torch.load(join(fold_dir, f"{mode}_pids.pt"))

    # Align using PIDs
    aligned_outcomes = outcomes.loc[pids]
    aligned_exposures = exposures.loc[pids]

    targets_array = aligned_outcomes.values
    exposures_array = aligned_exposures.values

    return TabularData(
        vectors=vectors, exposures=exposures_array, targets=targets_array, pids=pids
    )
