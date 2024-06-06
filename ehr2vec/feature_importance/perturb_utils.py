import logging
from os.path import join
from typing import Dict

import torch

from ehr2vec.data.batch import Batches

logger = logging.getLogger(__name__)


def average_sigmas(fi_folder:str, n_splits:int)->torch.Tensor:
    """
    Load and average sigmas from all folds. 
    Save sigmas_average.pt to fi_folder, return the averaged sigmas.
    """
    sigmas = []
    for fold in range(1, n_splits+1):
        sigmas_tensor = torch.load(join(fi_folder, f'sigmas_fold_{fold}.pt'))
        sigmas.append(sigmas_tensor)
    sigmas = torch.stack(sigmas).mean(dim=0)
    return sigmas


def compute_concept_frequency(features:Dict, vocabulary: Dict)->torch.Tensor:
    """Compute frequency of concepts in the features."""
    frequencies = torch.ones(len(vocabulary))
    concepts = torch.tensor(Batches.flatten(features['concept']))
    sorted_concepts, empiric_frequencies = torch.unique(concepts, return_counts=True)
    frequencies[sorted_concepts] += empiric_frequencies
    return frequencies