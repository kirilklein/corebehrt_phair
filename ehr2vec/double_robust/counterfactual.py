import logging
import random
from collections import Counter
from typing import Dict, List, Set, Optional

import numpy as np

from ehr2vec.common.utils import Data, iter_patients, match_pattern, match_patterns

logger = logging.getLogger(__name__)


class CounterfactualGenerator:
    def __init__(
        self,
        data: Data,
        exposure_regex: Optional[List[str]] = None,
        control_regex: Optional[str] = None,
    ):
        self.data = data
        self.exposure_regex = exposure_regex
        self.control_regex = control_regex
        self.exposure_codes: Set[int] = set()
        self.control_code: Optional[int] = None

    def generate(self) -> Data:
        """Main entry point to generate counterfactual data."""
        if self.exposure_regex is None:
            return self._handle_simple_exposure_flip()

        self._initialize_codes()
        return self._generate_counterfactual_data()

    def _initialize_codes(self) -> None:
        """Initialize exposure and control codes from regex patterns."""
        if not self.control_regex:
            raise ValueError("Control regex must be provided when using exposure regex")

        self.exposure_codes = match_patterns(self.exposure_regex, self.data.vocabulary)
        self.control_code = list(
            match_pattern(self.control_regex, self.data.vocabulary)
        )[0]

    def _generate_counterfactual_data(self) -> Data:
        """Generate counterfactual data by swapping codes."""
        code_frequencies = self._count_first_exposure_code_occurrence()
        exposure_code_probabilities = get_probability_of_codes(code_frequencies)

        counterfactual_features = self._generate_counterfactual_features(
            exposure_code_probabilities
        )

        counterfactual_exposures = self._get_counterfactual_exposures()

        return Data(
            features=counterfactual_features,
            pids=self.data.pids,
            outcomes=self.data.outcomes,
            vocabulary=self.data.vocabulary,
            exposed_patients=self._get_exposed_patients(counterfactual_exposures),
            exposures=counterfactual_exposures,
        )

    def _count_first_exposure_code_occurrence(
        self,
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
        for concept_list in self.data.features["concept"]:
            for code in concept_list:
                # If this code is an exposure code, increment and break
                if code in self.exposure_codes:
                    code_counts[code] += 1
                    break  # Only count the *first* exposure code in this list

        # Make sure every exposure_code is in the output, even if 0
        for code in self.exposure_codes:
            if code not in code_counts:
                code_counts[code] = 0

        return dict(code_counts)

    def _get_counterfactual_exposures(self) -> Optional[List[int]]:
        """Get counterfactual exposures by flipping all exposures."""
        return (
            [1 - exp for exp in self.data.exposures]
            if self.data.exposures is not None
            else None
        )

    def _generate_counterfactual_features(
        self, exposure_code_probabilities: Dict[int, float]
    ) -> Dict[str, List[List[int]]]:
        """Generate counterfactual features by swapping codes."""
        counterfactual_concepts = [
            swap_codes(
                patient["concept"], exposure_code_probabilities, self.control_code
            )
            for patient in iter_patients(self.data.features)
        ]

        counterfactual_features = self.data.features.copy()
        counterfactual_features["concept"] = counterfactual_concepts
        return counterfactual_features

    def _handle_simple_exposure_flip(self) -> Data:
        """Handle the simple case of flipping all exposures."""
        logger.info(
            "No exposure regex provided, flipping all exposures. Leave sequence unchanged."
        )
        counterfactual_exposures = [1 - exp for exp in self.data.exposures]

        return Data(
            features=self.data.features,
            pids=self.data.pids,
            outcomes=self.data.outcomes,
            vocabulary=self.data.vocabulary,
            exposed_patients=self._get_exposed_patients(counterfactual_exposures),
            exposures=counterfactual_exposures,
        )

    def _get_exposed_patients(self, exposures: Optional[List[int]]) -> List[str]:
        """Get list of exposed patient IDs based on exposures."""
        if not exposures:
            return []
        return [pid for pid, exp in zip(self.data.pids, exposures) if exp == 1]


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
