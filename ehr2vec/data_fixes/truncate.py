from typing import Dict, List

from ehr2vec.common.utils import iter_patients
from joblib import Parallel, delayed

class Truncator:
    def __init__(self, max_len: int, vocabulary: dict) -> None:
        self.max_len = max_len
        self.vocabulary = vocabulary
        self.sep_token = self.vocabulary.get("[SEP]")

    def __call__(self, features: List[Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
        return self.truncate(features)

    def truncate(self, features: List[Dict[str, List[str]]]) -> List[Dict[str, List[str]]]:
        background_length = self._get_background_length(features)


        def _process_patient(patient, background_length, truncator):
            return truncator._truncate_patient(patient, background_length)
        truncated_patients = Parallel(n_jobs=-1)(
            delayed(_process_patient)(patient, background_length, self) 
            for patient in features
        )
        return truncated_patients

    def _truncate_patient(self, patient: dict, background_length: int) -> dict:
        """Truncate patient to max_len, keeping background if present and CLS if present."""
        # Do not truncate if patient is shorter than max_len
        if len(patient["concept"]) <= self.max_len:
            return patient

        truncation_length = self.max_len - background_length

        # Do not start seq with [SEP] token (SEP token is included in background sentence)
        if patient["concept"][-truncation_length] == self.sep_token:
            truncation_length -= 1

        return {
            key: value[:background_length] + value[-truncation_length:]
            for key, value in patient.items()
        }

    def _get_background_length(self, features: List[Dict[str, List[str]]]) -> int:
        """Get the length of the background sentence, first SEP token included."""
        background_tokens = set(
            [v for k, v in self.vocabulary.items() if k.startswith("BG_")]
        )
        example_concepts = features[0]["concept"]
        cls_token_int = int(example_concepts[0] == self.vocabulary.get("[CLS]"))
        background_length = len(set(example_concepts) & background_tokens)

        return (
            background_length
            + int((background_length > 0) and self.sep_token is not None)
            + cls_token_int
        )
