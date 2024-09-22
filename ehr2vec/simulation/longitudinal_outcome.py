import numpy as np


def simulate_abspos_from_binary_outcome(
    binary_outcome: np.ndarray,
    index_dates: np.ndarray,
    days_offset: int = 0,
    max_years: int = 3,
) -> np.ndarray:
    """
    From a binary outcome, simulate absolute positions (times) of the outcomes.
    For 0s, the absolute position is set to None.
    For 1s, the absolute position is set to a random date between the index + days_offset date and the index date + max_years.
    """
    binary_outcome = binary_outcome.astype(bool)
    # Initialize the abspos array with None values
    abspos = np.full(binary_outcome.shape, None, dtype=object)

    # Generate random integers for the positions where outcome is 1
    abspos[binary_outcome] = np.random.randint(
        low=index_dates[binary_outcome] + days_offset * 24,
        high=index_dates[binary_outcome] + max_years * 365.25,
    )
    return abspos
