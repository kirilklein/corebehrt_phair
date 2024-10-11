import os
from os.path import join

import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_calibration(finetune_folder: str, method: str = "isotonic") -> None:
    predictions_df = pd.read_csv(join(finetune_folder, "predictions_and_targets.csv"))
    N_SPLITS: int = get_number_of_folds(finetune_folder)
    
    all_calibrated_predictions: list[pd.DataFrame] = []
    for fold in range(1, N_SPLITS + 1):
        fold_folder: str = join(finetune_folder, f"fold_{fold}")
        train_pids, val_pids = load_pids(fold_folder)
        train_data, val_data = split_data(predictions_df, train_pids, val_pids)
        
        calibrator = train_calibrator(train_data, method)
        calibrated_val_data: pd.DataFrame = calibrate_data(calibrator, val_data)
        all_calibrated_predictions.append(calibrated_val_data)

    combined_calibrated_df = pd.concat(all_calibrated_predictions, ignore_index=True)
    combined_calibrated_df.to_csv(join(finetune_folder, "predictions_and_targets_calibrated.csv"), index=False)

def get_number_of_folds(finetune_folder: str) -> int:
    return len([f for f in os.listdir(finetune_folder) if f.startswith("fold_")])

def load_pids(fold_folder: str) -> tuple[torch.Tensor, torch.Tensor]:
    train_pids: torch.Tensor = torch.load(join(fold_folder, "train_pids.pt"))
    val_pids: torch.Tensor = torch.load(join(fold_folder, "val_pids.pt"))
    return train_pids, val_pids

def split_data(predictions_df: pd.DataFrame, train_pids: torch.Tensor, val_pids: torch.Tensor) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data: pd.DataFrame = predictions_df[predictions_df['pid'].isin(train_pids)]
    val_data: pd.DataFrame = predictions_df[predictions_df['pid'].isin(val_pids)]
    return train_data, val_data

def train_calibrator(train_data: pd.DataFrame, method: str="isotonic") -> IsotonicRegression | LogisticRegression:
    """
    Train a calibrator for the given method.
    method{'isotonic', 'sigmoid'}, default='isotonic'
    """
    if method=="isotonic":
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif method=="sigmoid":
        calibrator = LogisticRegression()
    else:
        raise ValueError(f"Invalid calibration method: {method}")
    calibrator.fit(train_data['proba'].to_numpy().reshape(-1, 1), train_data['target'].to_numpy().ravel())
    return calibrator

def calibrate_data(calibrator, val_data: pd.DataFrame) -> pd.DataFrame:
    calibrated_probas = calibrator.predict(val_data['proba'].to_numpy().reshape(-1, 1))
    return val_data.assign(proba=calibrated_probas)

    
