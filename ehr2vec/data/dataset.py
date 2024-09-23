import logging

import pandas as pd
import torch
from ehr2vec.data.mask import ConceptMasker
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)  # Get the logger for this module


class BaseEHRDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features

    def _getpatient(self, index) -> dict:
        return {
            key: torch.as_tensor(values[index]) for key, values in self.features.items()
        }

    def __len__(self):
        return len(self.features["concept"])

    def __getitem__(self, index):
        return self._getpatient(index)


class MLMDataset(BaseEHRDataset):
    def __init__(
        self,
        features: dict,
        vocabulary: dict,
        select_ratio: float,
        masking_ratio: float = 0.8,
        replace_ratio: float = 0.1,
        ignore_special_tokens: bool = True,
    ):
        super().__init__(features)
        self.vocabulary = vocabulary
        self.masker = ConceptMasker(
            self.vocabulary,
            select_ratio,
            masking_ratio,
            replace_ratio,
            ignore_special_tokens,
        )

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        masked_concepts, target = self.masker.mask_patient_concepts(patient)
        patient["concept"] = masked_concepts
        patient["target"] = target
        return patient


class BinaryOutcomeDataset(BaseEHRDataset):
    """
    outcomes: absolute position when outcome occured for each patient
    outcomes is a list of the outcome timestamps to predict
    """

    def __init__(self, features: dict, outcomes: list):
        super().__init__(features)
        self.outcomes = outcomes

    def __getitem__(self, index: int) -> dict:
        patient = super().__getitem__(index)
        patient["target"] = float(pd.notna(self.outcomes[index]))

        return patient


class Time2EventDataset(BinaryOutcomeDataset):
    """
    outcomes: absolute position of outcome for each patient
    time2event: time to event for each patient
    """

    def __init__(self, features: dict, outcomes: list, time2event: list):
        super().__init__(features, outcomes)
        self.time2event = time2event

    def __getitem__(self, index: int) -> dict:
        """Add time2event to patient dict."""
        patient = super().__getitem__(index)
        patient["time2event"] = self.time2event[index]
        return patient
