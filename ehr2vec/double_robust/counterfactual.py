import random
from collections import Counter
from typing import Dict, List, Set

import numpy as np

from ehr2vec.common.utils import Data, iter_patients
from ehr2vec.data.utils import Utilities


def create_counterfactual_data(data: Data, exposure_regex_list: List[str]) -> Data:
    """
    Create counterfactual data by flipping the exposure variable.
    """
    exposure_codes = set()
    for exposure_regex in exposure_regex_list:
        exposure_codes.update(
            Utilities.get_codes_from_regex(data.vocabulary, exposure_regex)
        )

    code_frequencies = get_frequency_of_codes(data.features, exposure_codes)
    code_probabilities = get_probability_of_codes(code_frequencies)

    counterfactual_features = {key: [] for key in data.features}

    for patient in iter_patients(data.features):
        concepts = patient["concept"]
        if any(code in exposure_codes for code in concepts):
            patient = remove_codes(patient, exposure_codes)
        else:
            patient = insert_random_code_to_end(
                patient, exposure_codes, code_probabilities
            )
        for key, value in patient.items():
            counterfactual_features[key].append(value)
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


def remove_codes(patient: dict, codes: List[int]) -> dict:
    """
    Remove codes from patient sequences.
    """
    new_patient = {}
    indices = set([i for i, code in enumerate(patient["concept"]) if code in codes])
    for key, value in patient.items():
        new_patient[key] = [value[i] for i in range(len(value)) if i not in indices]
    return new_patient


def insert_random_code_to_end(
    patient: dict, exposure_codes: set, code_probabilities: Dict[int, float]
) -> dict:
    """
    Insert random code from exposure codes to the end of patient sequences. T
    """

    # make sure we have the correct order of the codes
    exposure_codes = list(exposure_codes)
    code_probabilities = [code_probabilities[code] for code in exposure_codes]

    new_patient = {}
    for key, value in patient.items():
        if key == "concept":
            new_patient[key] = value + [
                random.choices(exposure_codes, weights=code_probabilities)[0]
            ]
        else:
            new_patient[key] = value + [value[-1]]
    return new_patient

def insert_control_codes(data: Data, control_patients: set, control_code: str) -> Data:
    """Insert control codes for control patients at the closest event to the index date."""
    control_code = data.vocabulary[control_code]
    for i, patient in enumerate(iter_patients(data.features)):
        if data.pids[i] in control_patients:
            insert_control_code_for_patient(patient, data.index_dates[i], control_code)
    return data

def insert_control_code_for_patient(patient_data: dict, index_date: float, control_code: int) -> None:
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
        value = (control_code if key == "concept" 
                else index_date if key == "abspos"
                else patient_data[key][closest_event_idx])
        patient_data[key].insert(closest_event_idx, value)