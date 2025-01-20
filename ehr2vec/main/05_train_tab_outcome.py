"""
This script is used to train a tabular model on the patient vectors produced by the exposure model.
The goal is to restrict the features of the outcome model to the features of the exposure model (covariates).
"""

import os
from os.path import abspath, dirname, join
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from tqdm import tqdm

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.common.default_args import (
    DEFAULT_BLOBSTORE,
    ORG_PID_COL,
    PID_COL,
    PROBA_COL,
    TARGET_COL,
)
from ehr2vec.common.initialize import Initializer
from ehr2vec.common.loader import load_config
from ehr2vec.common.setup import DirectoryPreparer, get_args
from ehr2vec.downstream_tasks.outcomes import OutcomeHandler
from ehr2vec.evaluation.calibration import calibrate_data, train_calibrator

DEFAULT_CONFIG_NAME = "example_configs/05_train_tab_outcome.yaml"

args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


class VectorData:
    def __init__(self, vectors, exposures, targets, pids):
        self.vectors = vectors
        self.exposures = exposures
        self.targets = targets
        self.pids = pids


def load_vector_data(
    fold_path: str, mode: str, outcomes: pd.Series, exposures: pd.Series
) -> VectorData:
    """Load vectors, exposures, and targets for a specific fold and mode."""
    # Load vectors
    vectors = torch.load(join(fold_path, f"{mode}_patient_vectors.pt"))
    # Load PIDs
    pids = torch.load(join(fold_path, f"{mode}_pids.pt"))

    # Ensure everything is aligned using PIDs
    outcomes = outcomes.loc[pids]

    exposures = exposures.loc[pids]

    targets = outcomes.values
    exposures = exposures.values
    return VectorData(vectors, exposures, targets, pids)


def get_binary_outcomes(
    finetune_folder: str, outcome_path: str, n_hours_start_follow_up: int
) -> Tuple[pd.Series, pd.Series]:
    """Load and process binary outcome data.

    Args:
        finetune_folder: Path to folder containing finetuning results with index_dates.pt and pids.pt
        outcome_path: Path to CSV file containing outcome data
        n_hours_start_follow_up: Number of hours after index date to start follow-up period

    Returns:
        Tuple containing:
            - pd.Series: First outcome timestamps for each patient after follow-up start
            - pd.Series: Set of patient IDs that had outcomes before follow-up start
    """
    # Load index dates with weights_only=False since numpy dtype is needed
    index_dates = torch.load(
        join(finetune_folder, "index_dates.pt"), weights_only=False
    )
    pids = torch.load(join(finetune_folder, "pids.pt"), weights_only=True)
    index_dates = pd.Series(index_dates, index=pids)
    index_dates.index.name = PID_COL
    outcome_df = pd.read_csv(outcome_path)
    outcomes, outcome_pre_followup_pids = OutcomeHandler.get_first_outcome_in_follow_up(
        outcome_df, index_dates, n_hours_start_follow_up
    )
    # Create a row with 0 for every pid in index_dates that is not in outcomes
    outcomes = pd.Series(1, index=outcomes.index)
    outcomes.index.name = PID_COL

    missing_pids = set(index_dates.index) - set(outcomes.index)
    if missing_pids:
        missing_outcomes = pd.Series(0, index=list(missing_pids))
        missing_outcomes.index.name = PID_COL
        outcomes = pd.concat([outcomes, missing_outcomes])
    return outcomes, outcome_pre_followup_pids


def train_fold(
    cfg, fold_path: str, outcomes: pd.Series, exposures: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train XGBoost model on one fold"""
    # Load data
    train_data = load_vector_data(fold_path, "train", outcomes, exposures)
    val_data = load_vector_data(fold_path, "val", outcomes, exposures)

    # Prepare features (concatenate vectors with exposures if available)
    X_train = train_data.vectors.numpy()
    X_val = val_data.vectors.numpy()
    if train_data.exposures is not None:
        X_train = np.column_stack([X_train, train_data.exposures])
        X_val = np.column_stack([X_val, val_data.exposures])

    # Initialize and train XGBoost model
    xgb.set_config(verbosity=cfg.model.params.get("verbosity", 1))
    model = xgb.XGBClassifier(
        **cfg.model.params, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.fit(
        X_train,
        train_data.targets,
        eval_set=[(X_val, val_data.targets)],
    )

    # Validation predictions
    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    # Train calibrator if specified
    train_df = pd.DataFrame(
        data={
            PID_COL: train_data.pids,
            PROBA_COL: train_probs,
            TARGET_COL: train_data.targets,
        }
    )
    calibrator = train_calibrator(train_df, cfg.calibration)
    val_df = pd.DataFrame(
        data={
            PID_COL: val_data.pids,
            PROBA_COL: val_probs,
            TARGET_COL: val_data.targets,
        }
    )
    cal_val_df = calibrate_data(calibrator, val_df)
    # Validation predictions
    return val_df, cal_val_df


def main():
    # Load and process config
    cfg = load_config(config_path)
    override_config_from_cli(cfg)

    cfg, _, mount_context = Initializer.initialize_configuration_tabular(
        cfg, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
    )

    # Initialize folders and logging
    logger, xgboost_folder = DirectoryPreparer.setup_run_folder(cfg)
    cfg.save_to_yaml(join(xgboost_folder, "xgboost_config.yaml"))

    outcomes, outcome_pre_followup_pids = get_binary_outcomes(
        cfg.paths.model_path, cfg.paths.outcome, cfg.outcome.n_hours_start_follow_up
    )

    torch.save(
        outcome_pre_followup_pids, join(xgboost_folder, "outcome_pre_followup_pids.pt")
    )
    outcomes.to_csv(join(xgboost_folder, "binary_outcomes.csv"), index=False)
    # Load predictions and targets file to get exposures
    exposures = (
        pd.read_csv(join(cfg.paths.model_path, "predictions_and_targets.csv"))
        .rename(columns={ORG_PID_COL: PID_COL})[[PID_COL, TARGET_COL]]
        .set_index(PID_COL)
    )

    # Get number of folds from the finetune folder structure
    finetune_folder = cfg.paths.model_path
    fold_dirs = [d for d in os.listdir(finetune_folder) if d.startswith("fold_")]
    n_splits = len(fold_dirs)

    # Train for each fold
    result_dfs = []
    cal_result_dfs = []
    fold_loop = tqdm(fold_dirs, desc="Training folds")
    for fold_dir in fold_loop:
        fold = int(fold_dir.split("_")[1])
        logger.info(f"Training fold {fold}/{n_splits}")
        fold_path = join(finetune_folder, fold_dir)
        results, cal_results = train_fold(cfg, fold_path, outcomes, exposures)
        result_dfs.append(results)
        cal_result_dfs.append(cal_results)

    result_dfs = pd.concat(result_dfs)
    result_dfs = result_dfs.rename(columns={PID_COL: ORG_PID_COL})
    result_dfs.to_csv(join(xgboost_folder, f"predictions_and_targets.csv"), index=False)

    cal_result_dfs = pd.concat(cal_result_dfs)
    cal_result_dfs = cal_result_dfs.rename(columns={PID_COL: ORG_PID_COL})
    cal_result_dfs.to_csv(
        join(
            xgboost_folder, f"predictions_and_targets_calibrated_{cfg.calibration}.csv"
        ),
        index=False,
    )

    # Save to blob storage if using Azure
    if cfg.env == "azure":
        save_to_blobstore(
            local_path=cfg.paths.run_name,
            remote_path=join(cfg.get("project", DEFAULT_BLOBSTORE), cfg.paths.run_name),
        )
        mount_context.stop()

    logger.info("Done")


if __name__ == "__main__":
    main()
