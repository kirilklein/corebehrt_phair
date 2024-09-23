import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def get_concept_feature_importance(
    feature_importance: np.ndarray,
    vocabulary: Dict[str, int],
) -> Dict[str, float]:
    """Get feature importance of concept by name."""
    inv_vocab = {v: k for k, v in vocabulary.items()}
    return {inv_vocab[i]: importance for i, importance in enumerate(feature_importance)}


def log_most_important_features(
    feature_importance: np.ndarray, vocabulary: Dict[str, int], num_features: int = 20
) -> None:
    """Log the most important features based on the feature importance."""
    feature_importance_dic = get_concept_feature_importance(
        feature_importance, vocabulary
    )
    sorted_features = sort_dictionary(feature_importance_dic)
    sorted_features = sorted_features[:num_features]
    logger.info("Features with largest importance")
    for feature, importance in sorted_features:
        logger.info(f"{feature}: {importance}")


def log_most_important_features_deep(
    feature_importance: Dict[str, np.ndarray],
    vocabulary: Dict[str, int],
    num_features: int = 20,
) -> None:
    """Log the most important features based on the feature importance."""

    feature_importance_dic = get_concept_feature_importance(
        feature_importance["concept"], vocabulary
    )
    for k, v in feature_importance.items():
        if k != "concept":
            feature_importance_dic[k] = v[0]
    sorted_features = sort_dictionary(feature_importance_dic)
    sorted_features = sorted_features[:num_features]
    logger.info("Features with largest importance")
    for feature, importance in sorted_features:
        logger.info(f"{feature}: {importance}")


def sort_dictionary(dictionary: Dict[str, float]) -> List[Tuple[str, float]]:
    """Sort dictionary by values."""
    return sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
