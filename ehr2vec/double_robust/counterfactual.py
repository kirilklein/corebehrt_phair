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

    code_frequencies = count_first_exposure_code_occurrence(
        data.features, exposure_codes
    )
    logger.info(f"exposure code frequencies: {code_frequencies}")
    logger.info(f"sum of exposure code frequencies: {sum(code_frequencies.values())}")
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
    if data.exposures is not None:
        data.exposures = [1 - exp for exp in data.exposures]
    return Data(
        features=counterfactual_features,
        pids=data.pids,
        outcomes=data.outcomes,
        vocabulary=data.vocabulary,
        exposures=data.exposures,
    )


def get_probability_of_codes(
    code_frequencies: Dict[int, int],
) -> Dict[int, float]:
    """
    Get the probability of codes according to their frequency.
    """
    return {
        code: frequency / sum(code_frequencies.values())
        for code, frequency in code_frequencies.items()
    }


def count_first_exposure_code_occurrence(
    features: Dict[str, List[List[int]]], exposure_codes: Set[int]
) -> Dict[int, int]:
    """
    Get the frequency of exposure codes in the features, counting only the first
    occurrence of *any* exposure code in each concept_list.

    Args:
        features: Dictionary containing feature lists, including 'concept' key
                  with lists of concept codes.
        exposure_codes: Set of codes to count frequencies for.

    Returns:
        Dictionary mapping codes to their frequencies, where each concept_list
        contributes at most 1 count to exactly one code (the first code in the
        list that appears in exposure_codes).
    """
    code_counts = Counter()

    # Iterate over each concept_list
    for concept_list in features["concept"]:
        for code in concept_list:
            # If this code is an exposure code, increment and break
            if code in exposure_codes:
                code_counts[code] += 1
                break  # Only count the *first* exposure code in this list

    # Make sure every exposure_code is in the output, even if 0
    for code in exposure_codes:
        if code not in code_counts:
            code_counts[code] = 0

    return dict(code_counts)


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


def insert_control_codes(data: Data, control_patients: set, control_code: str) -> Data:
    """Insert control codes for control patients at the closest event to the index date."""
    control_code = data.vocabulary[control_code]
    for i, patient in enumerate(iter_patients(data.features)):
        if data.pids[i] in control_patients:
            insert_control_code_for_patient(patient, data.index_dates[i], control_code)
    return data


def insert_control_code_for_patient(
    patient_data: dict, index_date: float, control_code: int
) -> None:
    """
    Insert a control code and associated data into a patient's timeline at the event closest to their index date.

    Args:
        patient_data (dict): Dictionary containing patient timeline data with fields for concept codes,
            absolute positions (abspos), age, and segments
        index_date (float): The index date timestamp to insert the control code near
        control_code (int): The control code to insert into the patient timeline

    The function modifies the patient_data dictionary in-place by:
    1. Finding the event closest in time to the index_date
    2. Inserting the control_code at that position in the concept sequence
    3. Inserting corresponding values for abspos (index_date), age and segment
       (copied from closest event)
    """
    # Find event closest to index date
    closest_event_idx = np.abs(index_date - np.array(patient_data["abspos"])).argmin()

    # Insert control code and associated data at closest event position
    for key in patient_data.keys():
        value = (
            control_code
            if key == "concept"
            else index_date if key == "abspos" else patient_data[key][closest_event_idx]
        )
        patient_data[key].insert(closest_event_idx, value)
