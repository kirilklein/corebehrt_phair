import random
from typing import List

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

    counterfactual_features = {key: [] for key in data.features}

    for patient in iter_patients(data.features):
        concepts = patient["concept"]
        if any(code in exposure_codes for code in concepts):
            patient = remove_codes(patient, exposure_codes)
        else:
            patient = insert_random_code_to_end(patient, exposure_codes)
        for key, value in patient.items():
            counterfactual_features[key].append(value)
    return Data(
        counterfactual_features, data.outcomes, data.pids, data.mode, data.vocabulary
    )


def remove_codes(patient: dict, codes: List[int]) -> dict:
    """
    Remove codes from patient sequences.
    """
    new_patient = {}
    indices = set([i for i, code in enumerate(patient["concept"]) if code in codes])
    for key, value in patient.items():
        new_patient[key] = [value[i] for i in range(len(value)) if i not in indices]
    return new_patient


def insert_random_code_to_end(patient: dict, exposure_codes: set) -> dict:
    """
    Insert random code from exposure codes to the end of patient sequences. T
    """
    new_patient = {}
    for key, value in patient.items():
        if key == "concept":
            new_patient[key] = value + [random.choice(list(exposure_codes))]
        else:
            new_patient[key] = value + [value[-1]]
    return new_patient
