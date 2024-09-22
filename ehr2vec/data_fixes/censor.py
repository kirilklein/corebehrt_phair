from typing import List, Union, Dict

import pandas as pd

from ehr2vec.common.logger import TqdmToLogger
from ehr2vec.common.utils import iter_patients
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Censorer:
    def __init__(self, n_hours: int, n_hours_diag_censoring: int, vocabulary:dict=None, censor_diag_end_of_visit: bool=False) -> None:
        """Censor the features based on the event timestamp.
        n_hours if positive, censor all items that occur n_hours after event."""
        self.n_hours = n_hours
        self.vocabulary = vocabulary
        self.background_length = None
        self.n_hours_diag_censoring = n_hours_diag_censoring
        self.censor_diagnoses_at_end_of_visit = censor_diag_end_of_visit
        self.censor_diag_separately = self.censor_diagnoses_at_end_of_visit or (self.n_hours_diag_censoring!=self.n_hours)
        if self.censor_diag_separately:
            self.diagnoses_codes = self.get_diagnoses_codes()
            self.sep_code = self.vocabulary.get('[SEP]', -1)

    def __call__(self, features: dict, index_dates: list) -> tuple:
        sample_concepts = features["concept"][0]
        self.background_length = self.compute_background_length(sample_concepts)
        
        features = self.censor(features, index_dates)
        return features
    
    def get_diagnoses_codes(self) -> set:
        """
        Return the diagnoses codes.
        Codes starting with 'D'.
        """
        return set([v for k, v in self.vocabulary.items() if k.startswith('D') and (k!='Death')])

    def compute_background_length(self, sample_concepts) -> List[bool]:
        """Precompute background flags for the vocabulary."""
        tokenized_flag = self._identify_if_tokenized(sample_concepts)
        return sum(self._identify_background(sample_concepts, tokenized_flag))
    
    def censor(self, features: dict, index_dates: list) -> dict:
        """Censor the features based on the censor outcomes."""
        if not self.background_length:
            raise ValueError("Background length is not computed. Please call compute_background_length first.")
        censored_features = {key: [] for key in features}
        censor_loop = tqdm(iter_patients(features), desc='Censoring', 
            file=TqdmToLogger(logger), total=len(features["concept"]))
        for i, patient in enumerate(censor_loop):
            index_timestamp = index_dates[i]
            censored_patient = self._censor_patient(patient, index_timestamp)

            # Append to censored features
            for key, value in censored_patient.items():
                censored_features[key].append(value)
        return censored_features

    def _censor_patient(self, patient: dict, index_timestamp: float) -> dict:
        """Censor the patient's features n_hours after index_timestamp (given in abspos)."""
        if not pd.isna(index_timestamp):            
            censor_flags = self._generate_censor_flags(patient, index_timestamp)
            censor_flags[:self.background_length] = [True]*self.background_length # keep background
            for key, value in patient.items():
                patient[key] = [item for index, item in enumerate(value) if censor_flags[index]]
                
        return patient
    
    def _generate_censor_flags(self, patient: Dict[str, List], index_timestamp: float) -> List[bool]:
        """Generate flags indicating which items to censor, based on index_timestamp and self.n_hours."""
        absolute_positions = patient['abspos']
        censor_flags = [position  <= (index_timestamp + self.n_hours) for position in absolute_positions]
        if sum(censor_flags) == 0:
            return censor_flags
        if self.censor_diag_separately:
            if self.censor_diagnoses_at_end_of_visit:
                diag_censor_flags = self._generate_sep_diag_censor_flags_end_of_visit(patient, index_timestamp)
            else:
                diag_censor_flags = self._generate_sep_diag_censor_flags(patient, index_timestamp)
            censor_flags = self._combine_lists_with_or(censor_flags, diag_censor_flags)
        return censor_flags
    
    def _generate_sep_diag_censor_flags(self, patient: Dict[str, list], index_timestamp: float) -> List[bool]:
        """
        Censor diagnoses n_hours after the event.
        All diagnoses up to n_hours after index_timestamp are included.
        """
        diagnoses_sep_flags = self.get_combined_diag_sep_flags(patient['concept'])
        abs_pos_flags = [True if abspos <= (index_timestamp + self.n_hours_diag_censoring) else False for abspos in patient['abspos']]
        return self._combine_lists_with_and(diagnoses_sep_flags, abs_pos_flags)

    def _generate_sep_diag_censor_flags_end_of_visit(self, patient: Dict[str, list], index_timestamp: float) -> List[bool]:
        """
        Include diagnoses up to the end of the visit.
        Here, all diagnoses are included up to the end of the visit with index_timestamp.
        """
        last_segment = self._get_last_segment_before_timestamp(patient['segment'], patient['abspos'], index_timestamp)
        last_index_to_include = self._return_last_index_for_element(patient['segment'], last_segment)
        diagnoses_sep_flags = self.get_combined_diag_sep_flags(patient['concept'])
        return [
        i <= last_index_to_include and flag 
        for i, flag in enumerate(diagnoses_sep_flags)
            ]
    
    def get_combined_diag_sep_flags(self, concepts: List[Union[int, str]]) -> List[bool]:
        """
        Combining diagnoses and separator flags.
        Only SEP tokens after diagnoses are included.
        """
        diagnoses_flags = self._get_diagnoses_flags(concepts)
        sep_flags = self._get_sep_flags(concepts)
        return self._combine_diag_sep(diagnoses_flags, sep_flags)
    
    @staticmethod
    def _combine_diag_sep(diag_flags:List[bool], sep_flags:List[bool]) -> List[bool]:
        """
        Combining diagnoses and separator flags.
        We want to exclude SEP for visits with no diagnoses. Otherwise issues with the model (more visits than expected->indexing error in visit embeddings)
        example: 
        concepts = [D1, SEP, D2, D3, SEP, M1, SEP]
        diag_flags = [T, F, T, T, F, F, F]
        sep_flags = [F, T, F, F, T, F, T]
        result = [T, T, T, T, T, F, F]
        """
        result = []
        diag_seen = False  # Tracks if a diagnosis (True in diag_flags) has been seen since the last separator

        for diag, sep in zip(diag_flags, sep_flags):
            if diag:
                diag_seen = True  # A diagnosis has been seen, so keep the SEP flags
            result.append(diag or (sep and diag_seen))

            # Reset the diag_seen after a SEP, which is included in the result
            if sep and diag_seen:
                diag_seen = False

        return result

    def _get_sep_flags(self, concepts: List[int]) -> List[bool]:
        """This function returns a list of booleans indicating if the concept is a separator."""
        return [(concept == self.sep_code) for concept in concepts]

    @staticmethod
    def _combine_lists_with_and(list1: List[bool], list2: List[bool]) -> List[bool]:
        return [flag1 and flag2 for flag1, flag2 in zip(list1, list2)]
    
    @staticmethod
    def _combine_lists_with_or(list1: List[bool], list2: List[bool]) -> List[bool]:
        return [flag1 or flag2 for flag1, flag2 in zip(list1, list2)]

    def _get_diagnoses_flags(self, concept: List[Union[int, str]]) -> List[bool]:
        return [c in self.diagnoses_codes for c in concept]

    @staticmethod
    def _get_last_segment_before_timestamp(segments: List[int], abspos: List[float], index_timestamp: float)->int:
        """Return the last segment before the index_timestamp."""
        abspos_flags = [abspos <= index_timestamp for abspos in abspos]
        last_segment = [seg for seg, flag in zip(segments, abspos_flags) if flag][-1] 
        return last_segment
    
    @staticmethod
    def _return_last_index_for_element(lst: List[bool], element: bool)->int:
        """
        Return the max index of the element in the list.
        e.g. [1, 2, 3, 2, 1], 2 -> 3
        """
        return len(lst)-1-lst[::-1].index(element)

    def _identify_background(self, concepts: List[Union[int, str]], tokenized_flag: bool) -> List[bool]:
        """
        Identify background items in the patient's concepts.
        Return a list of booleans of the same length as concepts indicating if each item is background.
        """
        if tokenized_flag:
            bg_values = set([v for k, v in self.vocabulary.items() if k.startswith('BG_')])
            flags = [concept in bg_values for concept in concepts]
        else:
            flags = [concept.startswith('BG_') for concept in concepts]

        # Dont censor [CLS] and [SEP] tokens of background
        first_background = flags.index(True)
        if concepts[0] == '[CLS]' or concepts[0] == self.vocabulary.get('[CLS]'):
            flags[0] = True
        if concepts[first_background+1] == '[SEP]' or concepts[first_background+1] == self.vocabulary.get('[SEP]'):
            flags[first_background+1] = True

        return flags

    @staticmethod
    def _identify_if_tokenized(concepts:list) -> bool:
        """Identify if the features are tokenized."""
        return concepts and isinstance(concepts[0], int)

