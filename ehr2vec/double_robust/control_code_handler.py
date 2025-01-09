import numpy as np

from ehr2vec.common.utils import Data, iter_patients


def insert_control_codes(data: Data, control_patients: set, control_code: str) -> None:
    """Insert control codes for control patients at the closest event to their index date.

    This function processes a dataset and adds control codes for patients in the control group.
    For each control patient, it finds the event closest to their index date and inserts
    the control code at that position.

    Args:
        data: Data object containing patient features, IDs, index dates and vocabulary
        control_patients: Set of patient IDs that are in the control group
        control_code: String identifier for the control code in the vocabulary

    The function modifies the data object in-place by:
    1. Looking up the numeric control code from the vocabulary
    2. Iterating through all patients in the dataset
    3. For control patients, inserting the control code near their index date
    """
    control_code = data.vocabulary[control_code]
    for i, patient in enumerate(iter_patients(data.features)):
        if data.pids[i] in control_patients:
            insert_control_code_for_patient(patient, data.index_dates[i], control_code)


def insert_control_code_for_patient(
    patient_data: dict, index_date: float, control_code: int
) -> None:
    """
    Insert a control code and associated data into a patient's timeline at the appropriate position
    relative to the event closest to their index date.

    Args:
        patient_data (dict): Dictionary containing patient timeline data with fields for concept codes,
            absolute positions (abspos), age, and segments
        index_date (float): The index date timestamp to insert the control code near
        control_code (int): The control code to insert into the patient timeline

    The function modifies the patient_data dictionary in-place by:
    1. Finding the event closest in time to the index_date
    2. Determining whether to insert before or after based on timestamp comparison
    3. Inserting the control_code and associated data at the appropriate position
    """
    # Find event closest to index date
    closest_event_idx = _get_closest_event_idx(index_date, patient_data)

    # Determine insert position based on comparison with closest event
    insert_idx = closest_event_idx
    if index_date > patient_data["abspos"][closest_event_idx]:
        insert_idx += 1

    # Insert control code and associated data at determined position
    for key in patient_data:
        value = (
            control_code
            if key == "concept"
            else index_date if key == "abspos" else patient_data[key][closest_event_idx]
        )
        patient_data[key].insert(insert_idx, value)


def _get_closest_event_idx(index_date: float, patient_data: dict) -> int:
    """Get the index of the event closest to the index date."""
    return np.abs(index_date - np.array(patient_data["abspos"])).argmin()
