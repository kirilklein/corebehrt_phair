import logging
import os
import subprocess

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from ehr2vec.common.logger import TqdmToLogger

logger = logging.getLogger(__name__)  # Get the logger for this module


def get_nvidia_smi_output() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        return output
    except Exception as e:
        return str(e)


def get_tqdm(dataloader: DataLoader) -> tqdm:
    return tqdm(
        dataloader, total=len(dataloader), file=TqdmToLogger(logger) if logger else None
    )


def save_curves(
    run_folder: str, logits: torch.Tensor, targets: torch.Tensor, epoch: int, mode="val"
) -> None:
    """Saves the ROC and PRC curves to a csv file in the run folder"""
    roc_name = os.path.join(run_folder, "checkpoints", f"roc_curve_{mode}_{epoch}.npz")
    prc_name = os.path.join(run_folder, "checkpoints", f"prc_curve_{mode}_{epoch}.npz")
    probas = torch.sigmoid(logits).cpu().numpy()
    fpr, tpr, threshold_roc = roc_curve(targets, probas)
    precision, recall, threshold_pr = precision_recall_curve(targets, probas)
    np.savez_compressed(roc_name, fpr=fpr, tpr=tpr, threshold=threshold_roc)
    np.savez_compressed(
        prc_name,
        precision=precision,
        recall=recall,
        threshold=np.append(threshold_pr, 1),
    )


def save_predictions(
    run_folder: str, logits: torch.Tensor, targets: torch.Tensor, epoch: int, mode="val"
) -> None:
    """Saves the predictions to npz files in the run folder"""
    probas_name = os.path.join(run_folder, "checkpoints", f"probas_{mode}_{epoch}.npz")
    targets_name = os.path.join(
        run_folder, "checkpoints", f"targets_{mode}_{epoch}.npz"
    )
    probas = torch.sigmoid(logits).cpu().numpy()
    np.savez_compressed(probas_name, probas=probas)
    np.savez_compressed(targets_name, targets=targets)


def save_metrics_to_csv(run_folder: str, metrics: dict, epoch: int, mode="val") -> None:
    """Saves the metrics to a csv file"""
    metrics_name = os.path.join(run_folder, "checkpoints", f"{mode}_scores_{epoch}.csv")
    with open(metrics_name, "w") as file:
        file.write("metric,value\n")
        for key, value in metrics.items():
            file.write(f"{key},{value}\n")


def compute_avg_metrics(metric_values):
    """Compute average metrics."""
    avg_metrics = {}
    for name, values in metric_values.items():
        if isinstance(values[0], torch.Tensor):
            # Move tensors to CPU before converting to numpy
            values_array = torch.stack(values).cpu().numpy()
        else:
            values_array = np.array(values)
        
        # Check for NaN values
        if np.isnan(values_array).any():
            logger.info(f"Warning: NaN values detected in {name}")
            values_array = values_array[~np.isnan(values_array)]
        
        # Check for zero values
        if (values_array == 0).any():
            logger.info(f"Warning: Zero values detected in {name}")
        
        # Compute mean, avoiding division by zero
        if len(values_array) > 0:
            avg_metrics[name] = np.mean(values_array)
        else:
            logger.info(f"Warning: No valid values for {name}")
            avg_metrics[name] = np.nan

    return avg_metrics
