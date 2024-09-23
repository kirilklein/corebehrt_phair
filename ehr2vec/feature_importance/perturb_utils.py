import torch
from os.path import join
import logging
from typing import Dict
from ehr2vec.feature_importance.perturb import PerturbationModel
from ehr2vec.data.batch import Batches
from ehr2vec.feature_importance.utils import log_most_important_features

logger = logging.getLogger(__name__)


def average_sigmas(fi_folder: str, n_splits: int) -> torch.Tensor:
    """
    Load and average sigmas from all folds.
    Save sigmas_average.pt to fi_folder, return the averaged sigmas.
    """
    sigmas = []
    for fold in range(1, n_splits + 1):
        sigmas_tensor = torch.load(join(fi_folder, f"sigmas_fold_{fold}.pt"))
        sigmas.append(sigmas_tensor)
    sigmas = torch.stack(sigmas).mean(dim=0)
    return sigmas


def log_most_important_features_for_perturbation_model(
    perturbation_model: PerturbationModel,
    vocabulary: Dict[str, int],
    num_features: int = 20,
) -> None:
    """Log the most important features based on the sigmas from the perturbation model."""
    sigmas = perturbation_model.get_sigmas_weights()
    sigmas = sigmas.flatten().cpu().detach().numpy()
    feature_importance = 1 / (sigmas + 1e-9)
    log_most_important_features(feature_importance, vocabulary, num_features)


def compute_concept_frequency(features: Dict, vocabulary: Dict) -> torch.Tensor:
    """Compute frequency of concepts in the features."""
    frequencies = torch.ones(len(vocabulary))
    concepts = torch.tensor(Batches.flatten(features["concept"]))
    sorted_concepts, empiric_frequencies = torch.unique(concepts, return_counts=True)
    frequencies[sorted_concepts] += empiric_frequencies
    return frequencies
