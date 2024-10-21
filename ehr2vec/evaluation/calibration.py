import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def compute_and_save_calibration(
    finetune_folder: str, method: str = "isotonic"
) -> None:
    """
    Compute calibration for the predictions and save the results in a csv file: predictions_and_targets_calibrated_{method}.csv
    """
    predictions_df = pd.read_csv(join(finetune_folder, "predictions_and_targets.csv"))
    n_folds = get_number_of_folds(finetune_folder)

    all_calibrated_predictions: list[pd.DataFrame] = []
    for fold in range(1, n_folds + 1):
        fold_folder = join(finetune_folder, f"fold_{fold}")
        train_pids, val_pids = load_mode_pids("train", fold_folder), load_mode_pids(
            "val", fold_folder
        )
        train_data, val_data = split_data(predictions_df, train_pids, val_pids)

        calibrator: IsotonicRegression | LogisticRegression = train_calibrator(
            train_data, method
        )
        calibrated_val_data: pd.DataFrame = calibrate_data(calibrator, val_data)
        all_calibrated_predictions.append(calibrated_val_data)
        save_model(calibrator, fold_folder, method)

    combined_calibrated_df = pd.concat(all_calibrated_predictions, ignore_index=True)
    combined_calibrated_df.to_csv(
        join(finetune_folder, f"predictions_and_targets_calibrated_{method}.csv"),
        index=False,
    )


def calibrate_and_save_counterfactual_predictions(
    finetune_folder: str, counterfactual_folder: str, method: str
) -> None:
    """
    Calibrate counterfactual predictions for all folds.

    This function performs the following steps:
    1. Iterates over all folds
    2. Loads val PIDs for each fold
    3. Select val probas for the corresponding fold
    4. Calibrates the probabilities using the loaded model
    5. Accumulates the calibrated predictions
    6. Saves the combined calibrated predictions to a CSV file with PID and proba columns

    The calibrated predictions are saved in the counterfactual folder with the filename
    'counterfactual_predictions_calibrated_{method}.csv'.
    """
    n_folds = get_number_of_folds(finetune_folder)
    all_calibrated_predictions: list[pd.DataFrame] = []
    counterfactual_data: pd.DataFrame = pd.read_csv(
        join(counterfactual_folder, "counterfactual_predictions.csv")
    )

    for fold in range(1, n_folds + 1):
        fold_folder = join(finetune_folder, f"fold_{fold}")
        counterfactual_fold_folder = join(counterfactual_folder, f"fold_{fold}")

        val_pids = load_mode_pids("val", counterfactual_fold_folder)
        calibrator = load_model(fold_folder, method)

        _, val_data = split_data(counterfactual_data, set(), val_pids)
        calibrated_val_data = calibrate_data(calibrator, val_data)

        all_calibrated_predictions.append(calibrated_val_data)

    combined_calibrated_df = pd.concat(all_calibrated_predictions, ignore_index=True)
    combined_calibrated_df.to_csv(
        join(
            counterfactual_folder, f"counterfactual_predictions_calibrated_{method}.csv"
        ),
        index=False,
    )


def save_model(
    calibrator: IsotonicRegression | LogisticRegression, fold_folder: str, method: str
) -> None:
    """
    Save the calibrator to a pickle file.
    The file is named calibrator_{method}.pkl.
    """
    with open(join(fold_folder, f"calibrator_{method}.pkl"), "wb") as f:
        pickle.dump(calibrator, f)


def load_model(
    fold_folder: str, method: str
) -> IsotonicRegression | LogisticRegression:
    """Load the calibrator from a pickle file."""
    with open(join(fold_folder, f"calibrator_{method}.pkl"), "rb") as f:
        return pickle.load(f)


def get_number_of_folds(finetune_folder: str) -> int:
    """Get the number of folds in the finetune folder."""
    return len([f for f in os.listdir(finetune_folder) if f.startswith("fold_")])


def load_mode_pids(mode: str, fold_folder: str) -> torch.Tensor:
    """Load PIDs for the given mode from the given fold folder."""
    return torch.load(join(fold_folder, f"{mode}_pids.pt"))


def split_data(
    predictions_df: pd.DataFrame, train_pids: torch.Tensor, val_pids: torch.Tensor
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the predictions dataframe into train and val dataframes based on the given PIDs."""
    train_data: pd.DataFrame = predictions_df[predictions_df["pid"].isin(train_pids)]
    val_data: pd.DataFrame = predictions_df[predictions_df["pid"].isin(val_pids)]
    return train_data, val_data


def train_calibrator(
    train_data: pd.DataFrame, method: str = "isotonic"
) -> IsotonicRegression | LogisticRegression:
    """
    Train a calibrator for the given method.
    method{'isotonic', 'sigmoid'}, default='isotonic'
    """
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
    elif method == "sigmoid":
        calibrator = LogisticRegression()
    else:
        raise ValueError(f"Invalid calibration method: {method}")
    calibrator.fit(
        train_data["proba"].to_numpy().reshape(-1, 1),
        train_data["target"].to_numpy().ravel(),
    )
    return calibrator


def calibrate_data(
    calibrator: IsotonicRegression | LogisticRegression,
    val_data: pd.DataFrame,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Calibrate the probabilities of the given dataframe using the calibrator.
    Clip the probabilities to avoid values close to 0 or 1. (Often happening with isotonic regression)
    """
    calibrated_probas = calibrator.predict(val_data["proba"].to_numpy().reshape(-1, 1))
    calibrated_probas = np.clip(calibrated_probas, epsilon, 1 - epsilon)
    return val_data.assign(proba=calibrated_probas)
