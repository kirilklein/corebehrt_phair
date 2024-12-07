import logging
import random
from collections import Counter
from typing import Dict, List, Set

import numpy as np

from ehr2vec.common.utils import Data, iter_patients, match_pattern, match_patterns

logger = logging.getLogger(__name__)


def create_counterfactual_data(
    data: Data, exposure_regex: List[str], control_regex: str
) -> Data:
    """
    Create counterfactual data by flipping the exposure variable.
    """
    exposure_codes = match_patterns(exposure_regex, data.vocabulary)
    control_code = list(match_pattern(control_regex, data.vocabulary))[0]

    code_frequencies = get_frequency_of_codes(data.features, exposure_codes)
    logger.info(f"exposure code frequencies: {code_frequencies}")
    exposure_code_probabilities = get_probability_of_codes(code_frequencies)

    # Get counterfactual concepts by swapping codes for each patient
    counterfactual_concepts = []
    for patient in iter_patients(data.features):
        swapped_concepts = swap_codes(
            patient["concept"], exposure_code_probabilities, control_code
        )
        counterfactual_concepts.append(swapped_concepts)

    # Copy features and replace concept entry
    counterfactual_features = data.features.copy()
    counterfactual_features["concept"] = counterfactual_concepts

    return Data(
        features=counterfactual_features,
        pids=data.pids,
        outcomes=data.outcomes,
        vocabulary=data.vocabulary,
    )


def get_probability_of_codes(
    code_frequencies: Dict[int, int],
) -> Dict[int, float]:
    """
    Get the probability of codes according to their frequency.
    We add 1 to the denominator to avoid division by zero.
    """
    return {
        code: frequency / (sum(code_frequencies.values()) + 1)
        for code, frequency in code_frequencies.items()
    }


def get_frequency_of_codes(
    features: Dict[str, List[List[int]]], exposure_codes: Set[int]
) -> Dict[int, int]:
    """
    Get the frequency of codes in the features.

    Args:
        features: Dictionary containing feature lists, including 'concept' key with lists of concept codes
        exposure_codes: Set of codes to count frequencies for

    Returns:
        Dictionary mapping codes to their frequencies
    """
    # Flatten the list of concept lists and count only relevant codes
    all_concepts = [
        code
        for concept_list in features["concept"]
        for code in concept_list
        if code in exposure_codes
    ]

    # Use Counter for efficient counting
    code_counts = Counter(all_concepts)

    # Ensure all exposure codes are in the result, even if count is 0
    return {code: code_counts.get(code, 0) for code in exposure_codes}


def swap_codes(
    concepts: List[int],
    exposure_code_probabilities: Dict[int, float],
    control_code: int,
) -> List[int]:
    """
    Swap codes in concept field starting from the end of the sequence, stopping after first match.
    If exposure code present, replace it with control code.
    If control code present, draw exposure code with given probabilities.

    Args:
        concepts: List of concepts to swap
        exposure_code_probabilities: Dictionary of probabilities for each exposure code
        control_code: Control code to swap with
    Returns:
        List of new concepts
    """
    new_concepts = concepts.copy()

    exposure_codes = list(exposure_code_probabilities.keys())
    probs = [exposure_code_probabilities[code] for code in exposure_codes]
    # Process from end to start
    for i in range(len(concepts) - 1, -1, -1):
        code = concepts[i]
        if code in exposure_codes:
            new_concepts[i] = control_code
            break
        elif code == control_code:
            new_concepts[i] = random.choices(exposure_codes, weights=probs)[0]
            break

    return new_concepts
